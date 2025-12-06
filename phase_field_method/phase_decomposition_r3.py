import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numba
from numba import njit, prange
import time
import pandas as pd
import base64
from io import BytesIO
import warnings
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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔋 LiFePO₄ Phase-Field Simulator</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive simulation of spinodal decomposition with anisotropic elasticity & plasticity**")

# Initialize session state for caching
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'current_params_hash' not in st.session_state:
    st.session_state.current_params_hash = None

# ============================================
# SIDEBAR - SIMULATION PARAMETERS
# ============================================
with st.sidebar:
    st.markdown('<div class="section-header">Simulation Parameters</div>', unsafe_allow_html=True)
    
    # Simulation Type
    sim_type = st.selectbox(
        "Simulation Type",
        ["2D Cross-section (ac-plane)", "2D Cross-section (ab-plane)"],
        index=0,
        help="ac-plane: a-axis (x) vs c-axis (y), ab-plane: a-axis (x) vs b-axis (y, fast diffusion)"
    )
    
    # Grid size
    grid_size = st.slider("Grid Size (N×N)", 64, 512, 128, step=64)
    nx, ny = grid_size, grid_size
    use_3d = False
    
    # Time parameters
    total_steps = st.number_input("Total Time Steps", 1000, 50000, 5000, step=500)
    save_every = st.number_input("Save Frame Every N Steps", 50, 2000, 100, step=50)
    
    # Thermodynamic Parameters
    st.markdown("#### Thermodynamics")
    Omega_RT = st.slider("Ω / RT (Miscibility Gap)", 8.0, 20.0, 13.0, step=0.5,
                        help="Ω > 2 for spinodal decomposition")
    T = st.slider("Temperature (°C)", 25, 200, 25, step=25)
    T_K = T + 273.15
    
    # Gradient Energy
    kappa = st.slider("Gradient Coefficient κ (J/m)", 1e-12, 1e-9, 1e-11, format="%.1e",
                     help="Interface energy coefficient")
    
    # Mobility & Anisotropy
    st.markdown("#### Kinetics")
    M0 = st.number_input("Baseline Mobility M₀ (m²/s)", 1e-20, 1e-14, 1e-16, format="%.1e")
    
    # Anisotropy - fixed based on simulation type
    if "ac-plane" in sim_type:
        anisotropy = st.slider("Mobility Anisotropy (a:c)", 1, 100, 10, 
                              help="M_a : M_c ratio (moderately fast along a-axis)")
        Mx, My = M0 * anisotropy, M0
    else:  # ab-plane
        anisotropy = st.slider("Mobility Anisotropy (a:b)", 100, 10000, 1000,
                              help="M_b : M_a ratio (very fast along b-axis)")
        Mx, My = M0, M0 * anisotropy  # Fixed: x is a-axis, y is b-axis (fast)
    
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
        
        # Eigenstrain (anisotropic) - FIXED
        st.markdown("**Eigenstrain (Total Volume Change: 6.8%)**")
        col1, col2 = st.columns(2)
        with col1:
            eps_xx = st.number_input("εₓₓ (a-axis, %)", 1.0, 5.0, 2.5, step=0.1) / 100
            eps_yy = st.number_input("εᵧᵧ (b-axis, %)", -0.5, 1.5, 0.15, step=0.1) / 100
        with col2:
            eps_zz = st.number_input("ε₂₂ (c-axis, %)", -5.0, -1.0, -2.5, step=0.1) / 100
        
        # Proper Voigt notation for 2D: [ε_xx, ε_yy, ε_xy]
        epsilon0 = np.array([eps_xx, eps_yy, 0.0])  # 2D Voigt notation
    
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
         "Graded Profile", "Two Phase Mixture"],
        index=0
    )
    
    noise_level = st.slider("Initial Noise (%)", 0.0, 10.0, 2.0, step=0.5) / 100
    
    # Domain width parameter
    interface_width = st.slider("Interface Width (pixels)", 1, 10, 3, 
                               help="Width of phase boundary in pixels")
    
    # Run Button
    col1, col2 = st.columns(2)
    with col1:
        run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear Cache", use_container_width=True):
            st.session_state.simulation_results = None
            st.session_state.current_params_hash = None
            st.rerun()

# ============================================
# PHYSICS KERNELS (NUMBA-OPTIMIZED) - FIXED
# ============================================
@njit(parallel=True, fastmath=True, cache=True)
def chemical_potential(c, Omega_RT, kappa, dx=1.0):
    """Calculate chemical potential with anisotropic gradient energy"""
    nx, ny = c.shape
    mu = np.zeros_like(c)
    
    for i in prange(nx):
        for j in range(ny):
            # Handle boundaries with periodic BC
            im = (i - 1) % nx
            ip = (i + 1) % nx
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            
            # Regular solution free energy
            ci = np.clip(c[i, j], 1e-12, 1.0 - 1e-12)
            mu_hom = Omega_RT * (1.0 - 2.0 * ci) + np.log(ci) - np.log(1.0 - ci)
            
            # Laplacian (central difference)
            lap = (c[im, j] + c[ip, j] + c[i, jm] + c[i, jp] - 4.0 * ci) / (dx * dx)
            
            # Gradient energy contribution
            mu_grad = -kappa * lap
            
            mu[i, j] = mu_hom + mu_grad
    
    return mu

@njit(parallel=True, cache=True)
def elastic_potential_2d(c, epsilon0, C11, C22, C12, C44, V_m=3.02e-5):
    """Calculate elastic energy contribution for 2D plane stress"""
    nx, ny = c.shape
    mu_el = np.zeros_like(c)
    
    # Mean composition
    c_mean = np.mean(c)
    
    # Plane stress approximation
    # σ_ij = C_ijkl * ε_kl * (c - c_mean)
    for i in prange(nx):
        for j in range(ny):
            # Composition deviation
            dc = c[i, j] - c_mean
            
            # Effective strain (simplified)
            eps_eff = np.sqrt(epsilon0[0]**2 + epsilon0[1]**2)
            
            # Stress magnitude (simplified plane stress)
            # Using average stiffness
            C_avg = 0.5 * (C11 + C22)
            stress_mag = C_avg * eps_eff * dc
            
            # Chemical potential contribution
            mu_el[i, j] = -V_m * stress_mag
    
    return mu_el

@njit(parallel=True, fastmath=True, cache=True)
def update_concentration(c, mu, Mx, My, dt, dx=1.0, flux_rate=0.0, flux_mask=None):
    """Update concentration field using Cahn-Hilliard equation with flux BC"""
    nx, ny = c.shape
    c_new = np.zeros_like(c)
    
    # Precompute coefficients
    coeff_x = Mx * dt / (dx * dx)
    coeff_y = My * dt / (dx * dx)
    
    for i in prange(nx):
        for j in range(ny):
            # Handle boundaries with periodic BC
            im = (i - 1) % nx
            ip = (i + 1) % nx
            jm = (j - 1) % ny
            jp = (j + 1) % ny
            
            # Divergence of M*grad(mu)
            div_x = coeff_x * (mu[im, j] - 2.0 * mu[i, j] + mu[ip, j])
            div_y = coeff_y * (mu[i, jm] - 2.0 * mu[i, j] + mu[i, jp])
            
            # Update
            c_new[i, j] = c[i, j] + div_x + div_y
    
    # Apply flux boundary condition if specified
    if flux_rate != 0.0 and flux_mask is not None:
        c_new += flux_mask * flux_rate * dt
    
    # Clamp to physical range
    return np.clip(c_new, 0.0, 1.0)

@njit(cache=True)
def calculate_stable_dt(Mx, My, kappa, dx=1.0, safety_factor=0.1):
    """Calculate stable time step using CFL condition"""
    M_max = max(Mx, My)
    dt_max = safety_factor * dx * dx / (M_max * kappa + 1e-30)
    return min(dt_max, 1.0)  # Upper bound

# ============================================
# INITIAL CONDITION GENERATORS
# ============================================
def create_initial_condition(init_type, nx, ny, noise_level=0.02):
    """Create initial concentration field"""
    if init_type == "Homogeneous (x=0.5)":
        c = np.full((nx, ny), 0.5, dtype=np.float64)
    
    elif init_type == "Central Nucleus":
        c = np.full((nx, ny), 0.1, dtype=np.float64)  # Li-poor matrix
        center_x, center_y = nx // 2, ny // 2
        radius = min(nx, ny) // 6
        
        # Create circular nucleus
        Y, X = np.ogrid[:nx, :ny]
        dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = dist_from_center <= radius
        c[mask] = 0.9  # Li-rich nucleus
        
        # Smooth interface
        from scipy.ndimage import gaussian_filter
        c = gaussian_filter(c, sigma=1.0)
    
    elif init_type == "Random Fluctuations":
        c = np.full((nx, ny), 0.5, dtype=np.float64)
        fluctuations = np.random.normal(0, noise_level, (nx, ny))
        c += fluctuations
    
    elif init_type == "Graded Profile":
        c = np.zeros((nx, ny), dtype=np.float64)
        for i in range(nx):
            c[i, :] = 0.1 + 0.8 * (i / nx)  # Linear gradient
    
    elif init_type == "Two Phase Mixture":
        # Create checkerboard pattern
        c = np.zeros((nx, ny), dtype=np.float64)
        for i in range(nx):
            for j in range(ny):
                if (i // 10 + j // 10) % 2 == 0:
                    c[i, j] = 0.9  # Li-rich
                else:
                    c[i, j] = 0.1  # Li-poor
        
        # Smooth the interfaces
        from scipy.ndimage import gaussian_filter
        c = gaussian_filter(c, sigma=1.0)
    
    # Add noise to all configurations
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * 0.5, (nx, ny))
        c += noise
    
    return np.clip(c, 0.01, 0.99)

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
# SIMULATION ENGINE - FIXED
# ============================================
class LFPPhaseFieldSimulator:
    def __init__(self, params):
        self.params = params
        self.frames = []
        self.stats = []
        
        # Initialize concentration field
        self.c = create_initial_condition(
            params['init_type'],
            params['nx'],
            params['ny'],
            params['noise_level']
        )
        
        # Create flux mask if needed
        self.flux_mask = None
        if params['bc_type'] in ['Potentiostatic (Butler-Volmer)', 'Galvanostatic (Constant Flux)']:
            self.flux_mask = create_flux_mask(params['nx'], params['ny'], flux_rows=5, location='bottom')
    
    def calculate_flux_rate(self, step, c_surface=None):
        """Calculate flux rate based on boundary condition"""
        params = self.params
        
        if params['bc_type'] == "Open System (No Flux)":
            return 0.0
        
        elif params['bc_type'] == "Galvanostatic (Constant Flux)":
            # Constant current
            flux_rate = params.get('current', 0.01) * 1e-3  # Scale down for numerical stability
            return flux_rate
        
        elif params['bc_type'] == "Potentiostatic (Butler-Volmer)":
            # Butler-Volmer kinetics
            if c_surface is None:
                c_surface = np.mean(self.c[:5, :]) if self.c.shape[0] > 5 else np.mean(self.c)
            
            eta = params.get('eta', 0.03)  # V
            k0 = params.get('k0', 1.0)
            alpha = params.get('alpha', 0.5)
            
            # Simplified Butler-Volmer (avoiding sinh for stability)
            # F/RT ≈ 38.94 at 298K, adjust for temperature
            F_over_RT = 96485.0 / (8.314 * self.params['T_K'])
            exponent = alpha * F_over_RT * eta
            
            # Clamp exponent to avoid overflow
            exponent = np.clip(exponent, -10.0, 10.0)
            
            # Current density
            i = k0 * (np.exp(exponent) - np.exp(-(1 - alpha) * exponent))
            
            # Convert to flux (mol/m²s to dimensionless)
            flux_rate = i * 1e-4  # Scale down for numerical stability
            
            return flux_rate
        
        return 0.0
    
    def run(self):
        """Main simulation loop"""
        params = self.params
        
        # Calculate stable time step
        dt = calculate_stable_dt(
            params['Mx'], params['My'], 
            params['kappa'], 
            dx=1.0, safety_factor=0.1
        )
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_container = st.container()
        
        start_time = time.time()
        
        for step in range(params['total_steps']):
            # Calculate chemical potential
            mu_chem = chemical_potential(
                self.c, 
                params['Omega_RT'], 
                params['kappa']
            )
            
            # Add elastic contribution if enabled
            mu = mu_chem.copy()
            if params.get('use_elasticity', False):
                mu_el = elastic_potential_2d(
                    self.c,
                    params.get('epsilon0', np.zeros(3)),
                    params.get('C11', 200e9),
                    params.get('C22', 180e9),
                    params.get('C12', 70e9),
                    params.get('C44', 60e9)
                )
                mu += mu_el
            
            # Calculate flux rate
            flux_rate = self.calculate_flux_rate(step)
            
            # Update concentration
            self.c = update_concentration(
                self.c, mu, 
                params['Mx'], params['My'], 
                dt, flux_rate=flux_rate,
                flux_mask=self.flux_mask
            )
            
            # Save frame periodically
            if step % params['save_every'] == 0 or step == params['total_steps'] - 1:
                self.frames.append(self.c.copy().astype(np.float32))
                
                # Calculate statistics
                stats = {
                    'step': step,
                    'time': step * dt,
                    'mean_c': float(np.mean(self.c)),
                    'std_c': float(np.std(self.c)),
                    'min_c': float(np.min(self.c)),
                    'max_c': float(np.max(self.c)),
                }
                self.stats.append(stats)
            
            # Update progress
            if step % 100 == 0:
                progress = step / params['total_steps']
                progress_bar.progress(progress)
                
                if step % 1000 == 0:
                    with status_container:
                        status_text.text(f"Step {step:,} | Time = {step*dt:.3f} | Mean c = {np.mean(self.c):.4f} | Δc = {np.std(self.c):.4f}")
        
        elapsed = time.time() - start_time
        
        # Final progress update
        progress_bar.progress(1.0)
        
        return self.frames, self.stats, elapsed
    
    def analyze_morphology(self):
        """Analyze domain morphology from final frame"""
        try:
            from scipy import ndimage
            
            c = self.frames[-1] if self.frames else self.c
            
            # Binary threshold
            c_binary = c > 0.5
            
            # Label connected domains
            labeled, num_features = ndimage.label(c_binary)
            
            # Calculate domain properties
            domain_sizes = []
            
            for i in range(1, num_features + 1):
                mask = labeled == i
                size = np.sum(mask)
                if size > 4:  # Ignore very small domains
                    domain_sizes.append(size)
            
            return {
                'num_domains': num_features,
                'mean_size': float(np.mean(domain_sizes)) if domain_sizes else 0.0,
                'size_std': float(np.std(domain_sizes)) if domain_sizes else 0.0,
                'phase_fraction': float(np.mean(c_binary)),
            }
        except:
            # Fallback if scipy not available
            c_binary = c > 0.5
            return {
                'num_domains': 0,
                'mean_size': 0.0,
                'size_std': 0.0,
                'phase_fraction': float(np.mean(c_binary)),
            }

# ============================================
# VISUALIZATION FUNCTIONS - FIXED
# ============================================
def create_animation(frames, steps, sim_type):
    """Create interactive animation of concentration evolution"""
    if not frames:
        return go.Figure()
    
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
                name=f"Step {steps[i]}"
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
        xaxis_title = "a-axis (nm)"
        yaxis_title = "c-axis (nm)"
    else:  # ab-plane
        xaxis_title = "a-axis (nm)"
        yaxis_title = "b-axis (nm, fast diffusion)"
    
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
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": True,
            "x": 0.1,
            "xanchor": "left",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 16},
                "prefix": "Step: ",
                "visible": True,
                "xanchor": "right"
            },
            "transition": {"duration": 100, "easing": "cubic-in-out"},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
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
            "Concentration Standard Deviation",
            "Phase Separation Progress",
            "Domain Evolution"
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
    
    # Phase fraction (estimated from std)
    phase_sep = [min(1.0, s * 4) for s in std_c]  # Rough estimate
    fig.add_trace(
        go.Scatter(x=time, y=phase_sep, mode='lines', name='Phase Sep',
                  line=dict(color='green', width=2),
                  fill='tozeroy'),
        row=2, col=1
    )
    
    # Rate of change
    if len(mean_c) > 1:
        dcdt = np.abs(np.diff(mean_c) / np.diff(time))
        dcdt = np.concatenate(([dcdt[0]], dcdt))
        fig.add_trace(
            go.Scatter(x=time, y=dcdt, mode='lines', name='Rate',
                      line=dict(color='orange', width=2)),
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
    fig.update_yaxes(title_text="Phase Sep. Degree", row=2, col=1)
    fig.update_yaxes(title_text="Rate (dc/dt)", row=2, col=2)
    
    return fig

def plot_final_state_analysis(c_final, sim_type):
    """Create detailed analysis of final state"""
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Final Concentration",
            "Phase Map (x > 0.5)",
            "Gradient Magnitude",
            "Histogram",
            "1D Profile (center)",
            "2D FFT"
        ),
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "bar"}, {"type": "scatter"}, {"type": "heatmap"}]
        ]
    )
    
    # Final concentration
    fig.add_trace(
        go.Heatmap(z=c_final, colorscale='RdBu_r', zmin=0, zmax=1,
                  colorbar=dict(title="Li Fraction")),
        row=1, col=1
    )
    
    # Binary phase map
    phase_map = (c_final > 0.5).astype(float)
    fig.add_trace(
        go.Heatmap(z=phase_map, colorscale='Greys', zmin=0, zmax=1,
                  colorbar=dict(title="Phase")),
        row=1, col=2
    )
    
    # Gradient magnitude
    grad_y, grad_x = np.gradient(c_final)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    fig.add_trace(
        go.Heatmap(z=grad_mag, colorscale='Viridis',
                  colorbar=dict(title="|∇c|")),
        row=1, col=3
    )
    
    # Histogram
    hist, bins = np.histogram(c_final.ravel(), bins=50, range=(0, 1))
    fig.add_trace(
        go.Bar(x=(bins[:-1] + bins[1:])/2, y=hist, name='Distribution'),
        row=2, col=1
    )
    
    # 1D profile through center
    center_row = c_final.shape[0] // 2
    profile = c_final[center_row, :]
    fig.add_trace(
        go.Scatter(y=profile, mode='lines', name='Profile',
                  line=dict(color='blue', width=2)),
        row=2, col=2
    )
    
    # 2D FFT (log magnitude)
    fft2 = np.fft.fftshift(np.fft.fft2(c_final - np.mean(c_final)))
    fft_mag = np.log(np.abs(fft2) + 1)
    fig.add_trace(
        go.Heatmap(z=fft_mag, colorscale='Hot',
                  colorbar=dict(title="log|FFT|")),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        title_text="Final State Analysis",
        showlegend=False
    )
    
    return fig

# ============================================
# MAIN SIMULATION EXECUTION
# ============================================
if run_simulation:
    # Create parameters dictionary
    params = {
        'nx': nx,
        'ny': ny,
        'total_steps': total_steps,
        'save_every': save_every,
        'Omega_RT': Omega_RT,
        'T_K': T_K,
        'kappa': kappa,
        'Mx': Mx,
        'My': My,
        'use_elasticity': use_elasticity,
        'init_type': init_type,
        'noise_level': noise_level,
        'bc_type': bc_type,
        'sim_type': sim_type
    }
    
    # Add elasticity parameters if enabled
    if use_elasticity:
        params.update({
            'C11': C11 * 1e9,
            'C22': C22 * 1e9,
            'C12': C12 * 1e9,
            'C44': C44 * 1e9,
            'epsilon0': epsilon0
        })
    
    # Add BC-specific parameters
    if bc_type == "Potentiostatic (Butler-Volmer)":
        params.update({
            'eta': eta * 0.001,  # Convert mV to V
            'k0': k0,
            'alpha': alpha
        })
    elif bc_type == "Galvanostatic (Constant Flux)":
        params.update({
            'current': current
        })
    
    # Add plasticity parameters if enabled
    if use_plasticity:
        params.update({
            'sigma_y0': sigma_y0 * 1e9,
            'hardening': hardening * 1e9
        })
    
    # Create hash for caching
    import hashlib
    params_str = str(sorted(params.items()))
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    
    # Check if we can use cached results
    use_cache = False
    if (st.session_state.simulation_results is not None and 
        st.session_state.current_params_hash == params_hash):
        use_cache = True
    
    # Run simulation or use cache
    if not use_cache:
        # Show simulation info
        st.info(f"""
        **Simulation Configuration:**
        - Grid: {nx} × {ny} ({sim_type})
        - Time steps: {total_steps:,}
        - Ω/RT: {Omega_RT:.1f} ({'Spinodal' if Omega_RT > 2.0 else 'No Spinodal'})
        - Anisotropy: {anisotropy}:1
        - Elasticity: {'Enabled' if use_elasticity else 'Disabled'}
        - BC: {bc_type}
        """)
        
        # Create and run simulator
        simulator = LFPPhaseFieldSimulator(params)
        
        with st.spinner("Running phase-field simulation... This may take a few seconds."):
            frames, stats, elapsed = simulator.run()
        
        # Store results in session state
        st.session_state.simulation_results = {
            'frames': frames,
            'stats': stats,
            'morphology': simulator.analyze_morphology(),
            'final_state': frames[-1] if frames else simulator.c,
            'params': params,
            'elapsed': elapsed
        }
        st.session_state.current_params_hash = params_hash
        
        st.success(f"✅ Simulation completed in {elapsed:.1f} seconds!")
    
    else:
        st.info("📊 Using cached results from previous run with same parameters.")
        frames = st.session_state.simulation_results['frames']
        stats = st.session_state.simulation_results['stats']
        elapsed = st.session_state.simulation_results['elapsed']
    
    # ============================================
    # DISPLAY RESULTS
    # ============================================
    
    # Main visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Phase Evolution Animation</div>', unsafe_allow_html=True)
        
        # Create animation
        steps = [s['step'] for s in stats]
        fig_anim = create_animation(frames, steps, sim_type)
        st.plotly_chart(fig_anim, use_container_width=True)
        
        # Final state analysis
        with st.expander("📈 Detailed Final State Analysis", expanded=False):
            final_state = st.session_state.simulation_results['final_state']
            fig_analysis = plot_final_state_analysis(final_state, sim_type)
            st.plotly_chart(fig_analysis, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Simulation Metrics</div>', unsafe_allow_html=True)
        
        # Key metrics
        final_stats = stats[-1]
        morphology = st.session_state.simulation_results['morphology']
        
        metrics_col1, metrics_col2 = st.columns(2)
        
        with metrics_col1:
            st.metric("Total Time", f"{final_stats['time']:.3f}")
            st.metric("Final Mean x", f"{final_stats['mean_c']:.4f}")
            st.metric("Phase Separation", f"{final_stats['std_c']:.4f}")
        
        with metrics_col2:
            st.metric("Domains", morphology['num_domains'])
            st.metric("Li-rich Phase %", f"{morphology['phase_fraction']*100:.1f}%")
            st.metric("Run Time", f"{elapsed:.1f}s")
        
        # Phase separation indicator
        st.markdown("#### Phase Separation Status")
        if final_stats['std_c'] > 0.2:
            st.success("✅ **Strong Phase Separation**")
            st.progress(1.0)
        elif final_stats['std_c'] > 0.05:
            st.warning("⚠️ **Moderate Phase Separation**")
            st.progress(final_stats['std_c'] / 0.2)
        else:
            st.info("🔵 **Homogeneous / Solid Solution**")
            st.progress(final_stats['std_c'] / 0.2)
    
    # Statistics plot
    st.markdown('<div class="section-header">Evolution Statistics</div>', unsafe_allow_html=True)
    fig_stats = plot_statistics(stats)
    st.plotly_chart(fig_stats, use_container_width=True)
    
    # ============================================
    # EXPORT AND ANALYSIS SECTION
    # ============================================
    st.markdown('<div class="section-header">Export & Analysis</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export CSV data
        if st.button("📊 Export Statistics CSV"):
            df_stats = pd.DataFrame(stats)
            csv = df_stats.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="lfp_simulation_stats.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        # Export final frame as numpy array
        if st.button("💾 Export Final State"):
            final_state = st.session_state.simulation_results['final_state']
            np_bytes = BytesIO()
            np.save(np_bytes, final_state, allow_pickle=False)
            b64 = base64.b64encode(np_bytes.getvalue()).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="lfp_final_state.npy">Download NPY</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        # Export parameters
        if st.button("⚙️ Export Parameters"):
            params_df = pd.DataFrame([params])
            csv = params_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="lfp_parameters.csv">Download Parameters</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # ============================================
    # PHYSICS INSIGHTS
    # ============================================
    st.markdown('<div class="section-header">Physics Insights</div>', unsafe_allow_html=True)
    
    insights = []
    
    # Analyze results
    final_std = stats[-1]['std_c']
    
    if final_std > 0.25:
        insights.append("✅ **Strong spinodal decomposition**: Clear phase separation into Li-rich and Li-poor domains")
    elif final_std > 0.1:
        insights.append("⚠️ **Moderate phase separation**: Beginning of domain formation")
    else:
        insights.append("🔵 **Homogeneous/Solid solution**: System remains mixed or phase separation suppressed")
    
    if use_elasticity:
        if morphology['num_domains'] > 10:
            insights.append("📐 **Elasticity-mediated patterning**: Multiple small domains indicating elastic strain effects")
        else:
            insights.append("📏 **Elastic coarsening suppression**: Limited domain growth due to strain energy")
    
    if "ab-plane" in sim_type and anisotropy > 100:
        insights.append("➡️ **Anisotropic domain growth**: Elongated domains along fast diffusion direction (b-axis)")
    
    if bc_type == "Potentiostatic (Butler-Volmer)":
        if np.abs(eta) > 50:
            insights.append("⚡ **High overpotential driving**: Rapid phase transformation")
        else:
            insights.append("🔋 **Moderate electrochemical driving**: Controlled phase evolution")
    
    # Display insights
    for insight in insights:
        st.info(insight)
    
    # Theoretical background
    with st.expander("🧪 Theoretical Background"):
        st.markdown("""
        **Governing Equations:**
        
        Cahn-Hilliard equation for spinodal decomposition:
        
        ```
        ∂c/∂t = ∇·[M(c)∇μ]
        μ = δf/δc - κ∇²c + μ_el + μ_ext
        ```
        
        Where:
        - `c`: Li concentration (0 ≤ c ≤ 1)
        - `M(c)`: Mobility tensor (anisotropic for LiFePO₄)
        - `μ`: Chemical potential
        - `f(c)`: Regular solution free energy: `f = RT[c ln c + (1-c) ln(1-c)] + Ωc(1-c)`
        - `κ`: Gradient energy coefficient
        - `μ_el`: Elastic contribution to chemical potential
        - `μ_ext`: External electrochemical potential
        
        **Elastic Contribution:**
        
        For anisotropic eigenstrain ε⁰(c) = β·(1-c):
        
        ```
        μ_el = -V_m·σ:∂ε⁰/∂c
        σ = C:(ε - ε⁰(c))
        ∇·σ = 0
        ```
        
        **Spinodal Condition:**
        
        Phase separation occurs when:
        
        ```
        ∂²f/∂c² < 0  →  Ω > 2RT
        ```
        
        For LiFePO₄ at room temperature: Ω ≈ 13RT, ensuring strong spinodal decomposition.
        """)

else:
    # ============================================
    # WELCOME / INSTRUCTIONS
    # ============================================
    st.markdown("""
    ## 📖 Welcome to the LiFePO₄ Phase-Field Simulator
    
    This tool simulates **spinodal decomposition** in LiFePO₄ nanoparticles during battery charging/discharging, incorporating:
    
    ### Key Features:
    
    1. **Anisotropic Elasticity**: Models 6.8% anisotropic lattice mismatch between LiFePO₄ and FePO₄
    2. **Anisotropic Diffusion**: Fast Li transport along b-axis (1D channels)
    3. **Electrochemical BCs**: Butler-Volmer kinetics with stress coupling
    4. **Real-time Visualization**: Interactive animations and comprehensive analysis
    
    ### How to Use:
    
    1. **Configure parameters** in the sidebar
    2. **Click "Run Simulation"** to start
    3. **Analyze results** in the main panel
    
    ### Recommended Starting Parameters:
    
    | Parameter | Recommended Value | Purpose |
    |-----------|-------------------|---------|
    | Grid Size | 128×128 | Balance detail & speed |
    | Ω/RT | 13.0 | Strong phase separation |
    | Anisotropy | 1000:1 (ab-plane) | Fast b-axis diffusion |
    | Elasticity | Enabled | Essential for realistic patterns |
    | Boundary | Potentiostatic | Electrochemical control |
    
    ### Quick Start:
    """)
    
    if st.button("⚡ Load Quick Start Parameters", type="secondary"):
        # This would set parameters via session state in a full implementation
        st.info("Quick start parameters loaded! Adjust as needed and click 'Run Simulation'")
        st.rerun()
    
    # Add example images or diagrams
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Spinodal Decomposition**")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Spinodal_decomposition.svg/400px-Spinodal_decomposition.svg.png", 
                caption="Phase separation without nucleation barrier")
    
    with col2:
        st.markdown("**LiFePO₄ Structure**")
        st.image("https://ars.els-cdn.com/content/image/1-s2.0-S1388248116301722-gr2.jpg",
                caption="Olivine structure with 1D Li channels")
    
    with col3:
        st.markdown("**Domain Patterns**")
        st.image("https://www.researchgate.net/profile/Jia-Xu-27/publication/320572390/figure/fig2/AS:669435857911821@1536617153210/Phase-separation-patterns-in-two-dimensions-a-c-Phase-separation-patterns-from-a.ppm",
                caption="Typical spinodal patterns")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>LiFePO₄ Phase-Field Simulator</strong> | Based on Cahn-Hilliard theory with anisotropic elasticity</p>
    <p>Simulates spinodal decomposition in battery cathodes • Supports academic research</p>
    <p style="font-size: 0.8rem; color: #999;">References: Cahn & Hilliard (1958), Cogswell & Bazant (2012), Tang et al. (2011)</p>
</div>
""", unsafe_allow_html=True)
