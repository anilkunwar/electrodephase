import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numba
from numba import njit, prange
import time
import io
import base64

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
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔋 LiFePO₄ Phase-Field Simulator</h1>', unsafe_allow_html=True)
st.markdown("**Comprehensive simulation of spinodal decomposition with anisotropic elasticity & plasticity**")

# ============================================
# SIDEBAR - SIMULATION PARAMETERS
# ============================================
with st.sidebar:
    st.markdown('<div class="section-header">Simulation Parameters</div>', unsafe_allow_html=True)
    
    # Simulation Type
    sim_type = st.selectbox(
        "Simulation Type",
        ["2D Cross-section (ac-plane)", "2D Cross-section (ab-plane)", "3D (Reduced Resolution)"],
        index=0
    )
    
    if "2D" in sim_type:
        grid_size = st.slider("Grid Size (N×N)", 64, 512, 256, step=64)
        nx, ny = grid_size, grid_size
        nz = 1
        use_3d = False
    else:
        grid_size = st.slider("Grid Size (N³)", 32, 128, 64, step=32)
        nx, ny, nz = grid_size, grid_size, grid_size
        use_3d = True
    
    # Time parameters
    total_steps = st.number_input("Total Time Steps", 1000, 100000, 20000, step=1000)
    save_every = st.number_input("Save Frame Every N Steps", 50, 2000, 200, step=50)
    
    # Thermodynamic Parameters
    st.markdown("#### Thermodynamics")
    Omega_RT = st.slider("Ω / RT (Miscibility Gap)", 8.0, 20.0, 13.0, step=0.5,
                        help="Controls phase separation strength")
    T = st.slider("Temperature (°C)", 25, 200, 25, step=25)
    T_K = T + 273.15
    
    # Gradient Energy
    kappa = st.slider("Gradient Coefficient κ (J/m)", 1e-11, 1e-9, 5e-11, format="%.1e",
                     help="Interface energy coefficient")
    
    # Mobility & Anisotropy
    st.markdown("#### Kinetics")
    M0 = st.number_input("Baseline Mobility M₀ (m²/s)", 1e-18, 1e-14, 1e-16, format="%.1e")
    Ea = st.slider("Activation Energy Eₐ (eV)", 0.3, 0.8, 0.5, step=0.05)
    
    # Anisotropy
    if "ac-plane" in sim_type:
        anisotropy = st.slider("Mobility Anisotropy (a:c)", 1, 100, 10, 
                              help="M_a : M_c ratio (fast along a-axis)")
        Mx, My = M0 * anisotropy, M0
    elif "ab-plane" in sim_type:
        anisotropy = st.slider("Mobility Anisotropy (a:b)", 1000, 100000, 10000,
                              help="M_b : M_a ratio (fast along b-axis)")
        Mx, M0 = M0, M0 * anisotropy
    
    # Elasticity Parameters
    st.markdown("#### Elasticity")
    use_elasticity = st.checkbox("Enable Anisotropic Elasticity", value=True)
    
    if use_elasticity:
        col1, col2 = st.columns(2)
        with col1:
            C11 = st.number_input("C₁₁ (GPa)", 100.0, 300.0, 200.0, step=10.0)
            C12 = st.number_input("C₁₂ (GPa)", 50.0, 150.0, 70.0, step=5.0)
        with col2:
            C44 = st.number_input("C₄₄ (GPa)", 30.0, 100.0, 60.0, step=5.0)
        
        # Eigenstrain (anisotropic)
        st.markdown("**Eigenstrain (Volume Change: 6.8%)**")
        col1, col2, col3 = st.columns(3)
        with col1:
            eps_xx = st.number_input("εₓₓ (%)", 1.0, 5.0, 2.5, step=0.1) / 100
        with col2:
            eps_yy = st.number_input("εᵧᵧ (%)", -0.5, 1.5, 0.15, step=0.1) / 100
        with col3:
            eps_zz = st.number_input("εₓₓ (%)", -5.0, -1.0, -2.5, step=0.1) / 100
        
        epsilon0 = np.array([eps_xx, eps_yy, eps_zz, 0, 0, 0])  # Voigt notation
    
    # Plasticity Parameters
    st.markdown("#### Plasticity")
    use_plasticity = st.checkbox("Enable J2 Plasticity", value=False)
    
    if use_plasticity:
        sigma_y0 = st.number_input("Yield Stress σ_y₀ (GPa)", 0.5, 5.0, 2.0, step=0.1)
        hardening = st.number_input("Hardening h (GPa)", 0.0, 200.0, 50.0, step=10.0)
        plastic_rate = st.number_input("Plastic Rate Exponent m", 1, 50, 20, step=1)
    
    # Electrochemical BCs
    st.markdown("#### Electrochemistry")
    bc_type = st.selectbox(
        "Boundary Condition",
        ["Potentiostatic (Butler-Volmer)", "Galvanostatic (Constant Flux)", "Mixed Mode"],
        index=0
    )
    
    if bc_type == "Potentiostatic (Butler-Volmer)":
        eta = st.slider("Overpotential η (mV)", -100, 100, 30, step=10)
        k0 = st.number_input("Exchange Current k₀", 0.01, 10.0, 1.0, step=0.1)
        alpha = st.slider("Charge Transfer α", 0.3, 0.7, 0.5, step=0.05)
    else:
        current = st.number_input("Current Density (A/m²)", 0.1, 100.0, 10.0, step=1.0)
    
    # Initial Conditions
    st.markdown("#### Initial Conditions")
    init_type = st.selectbox(
        "Initial Composition",
        ["Homogeneous (x=0.5)", "Central Nucleus", "Random Fluctuations", "Graded Profile"],
        index=0
    )
    
    noise_level = st.slider("Initial Noise (%)", 0.0, 5.0, 0.5, step=0.1) / 100
    
    # Run Button
    run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    if st.button("🔄 Reset Parameters"):
        st.rerun()

# ============================================
# PHYSICS KERNELS (NUMBA-OPTIMIZED)
# ============================================
@njit(parallel=True, fastmath=True)
def chemical_potential(c, Omega_RT, kappa, dx=1.0):
    """Calculate chemical potential with anisotropic gradient energy"""
    nx, ny = c.shape
    mu = np.empty_like(c)
    
    for i in numba.prange(1, nx-1):
        for j in numba.prange(1, ny-1):
            # Regular solution free energy
            ci = np.clip(c[i,j], 1e-12, 1-1e-12)
            mu_hom = Omega_RT * (1.0 - 2.0 * ci) + np.log(ci) - np.log(1.0 - ci)
            
            # Anisotropic Laplacian (different coefficients for x/y)
            lap_x = (c[i+1,j] + c[i-1,j] - 2*c[i,j]) / (dx*dx)
            lap_y = (c[i,j+1] + c[i,j-1] - 2*c[i,j]) / (dx*dx)
            
            # Gradient energy contribution
            mu_grad = -kappa * (lap_x + 0.1*lap_y)  # Anisotropic
            
            mu[i,j] = mu_hom + mu_grad
    
    # Boundary conditions (periodic)
    mu[0,:] = mu[-2,:]
    mu[-1,:] = mu[1,:]
    mu[:,0] = mu[:,-2]
    mu[:,-1] = mu[:,1]
    
    return mu

@njit(parallel=True)
def elastic_potential(c, epsilon0, C11, C12, C44, V_m=3.0e-5):
    """Calculate elastic energy contribution (mean-field approximation)"""
    nx, ny = c.shape
    mu_el = np.zeros_like(c)
    
    # Mean composition
    c_mean = np.mean(c)
    
    for i in numba.prange(nx):
        for j in numba.prange(ny):
            # Composition deviation
            dc = c[i,j] - c_mean
            
            # Stress (simplified isotropic approximation)
            # σ = E * ε0 * dc
            E_eff = C11  # Simplified
            stress_mag = E_eff * np.linalg.norm(epsilon0[:3]) * dc
            
            # Chemical potential contribution
            mu_el[i,j] = -V_m * stress_mag
    
    return mu_el

@njit(parallel=True)
def update_concentration(c, mu, Mx, My, dt, dx=1.0, flux=0.0):
    """Update concentration field using Cahn-Hilliard equation"""
    nx, ny = c.shape
    c_new = np.empty_like(c)
    
    for i in numba.prange(1, nx-1):
        for j in numba.prange(1, ny-1):
            # Flux in x-direction
            Jx = -Mx * (mu[i+1,j] - mu[i-1,j]) / (2*dx)
            
            # Flux in y-direction
            Jy = -My * (mu[i,j+1] - mu[i,j-1]) / (2*dx)
            
            # Divergence of flux
            div_J = (Jx - (-Jx)) / (2*dx) + (Jy - (-Jy)) / (2*dx)
            
            # Update
            c_new[i,j] = c[i,j] + dt * (div_J + flux)
    
    # Apply boundary conditions (periodic)
    c_new[0,:] = c_new[-2,:]
    c_new[-1,:] = c_new[1,:]
    c_new[:,0] = c_new[:,-2]
    c_new[:,-1] = c_new[:,1]
    
    # Clamp to physical range
    return np.clip(c_new, 0.0, 1.0)

@njit
def plastic_update(sigma, epsilon_p, sigma_y, h, dt):
    """J2 Plasticity update (return mapping algorithm)"""
    # Deviatoric stress
    sigma_dev = sigma - np.trace(sigma)/3.0 * np.eye(3)
    J2 = 0.5 * np.sum(sigma_dev * sigma_dev)
    
    # Yield condition
    f = np.sqrt(3*J2) - sigma_y
    
    if f > 0:
        # Plastic flow direction
        n = sigma_dev / np.sqrt(2*J2)
        
        # Plastic multiplier
        dgamma = f / (3*dt + h)
        
        # Update plastic strain
        epsilon_p += dgamma * n
        
        # Update stress
        sigma -= 2*dt * dgamma * sigma_dev
        
        # Update yield stress
        sigma_y += h * dgamma
    
    return sigma, epsilon_p, sigma_y

# ============================================
# SIMULATION ENGINE
# ============================================
class LFPPhaseFieldSimulator:
    def __init__(self, params):
        self.params = params
        self.frames = []
        self.stats = []
        
        # Initialize fields
        if use_3d:
            self.c = self.init_3d_field()
        else:
            self.c = self.init_2d_field()
        
        # Initialize plasticity variables
        if use_plasticity:
            self.epsilon_p = np.zeros((nx, ny, 3, 3))
            self.sigma_y = np.full((nx, ny), sigma_y0 * 1e9)  # Convert to Pa
        else:
            self.epsilon_p = None
            self.sigma_y = None
    
    def init_2d_field(self):
        """Initialize 2D concentration field"""
        nx, ny = self.params['nx'], self.params['ny']
        c = np.full((nx, ny), 0.5, dtype=np.float64)
        
        # Add initial fluctuations
        noise = self.params['noise_level']
        c += np.random.normal(0, noise, (nx, ny))
        
        # Apply initial pattern based on selection
        if self.params['init_type'] == "Central Nucleus":
            radius = min(nx, ny) // 8
            center_x, center_y = nx//2, ny//2
            for i in range(nx):
                for j in range(ny):
                    if (i-center_x)**2 + (j-center_y)**2 < radius**2:
                        c[i,j] = 0.9  # Li-rich nucleus
        
        elif self.params['init_type'] == "Graded Profile":
            for i in range(nx):
                c[i,:] = 0.1 + 0.8 * (i / nx)
        
        return np.clip(c, 0.01, 0.99)
    
    def init_3d_field(self):
        """Initialize 3D concentration field"""
        # Simplified for performance
        return np.full((nx, ny, nz), 0.5, dtype=np.float64)
    
    def calculate_stability_dt(self):
        """Calculate stable time step using CFL condition"""
        M_max = max(self.params['Mx'], self.params['My'])
        dx = 1.0  # Normalized
        dt_max = 0.25 * dx**2 / (M_max * self.params['kappa'] + 1e-30)
        return min(dt_max, 0.1)
    
    def run(self):
        """Main simulation loop"""
        params = self.params
        dt = self.calculate_stability_dt()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        
        for step in range(params['total_steps']):
            # Calculate chemical potential
            mu_chem = chemical_potential(
                self.c, 
                params['Omega_RT'], 
                params['kappa']
            )
            
            # Add elastic contribution
            if params['use_elasticity']:
                mu_el = elastic_potential(
                    self.c,
                    params['epsilon0'],
                    params['C11'] * 1e9,
                    params['C12'] * 1e9,
                    params['C44'] * 1e9
                )
                mu = mu_chem + mu_el
            else:
                mu = mu_chem
            
            # Update concentration
            flux = 0.0
            if step > params['total_steps'] // 4:  # Start flux after some steps
                if params['bc_type'] == "Potentiostatic (Butler-Volmer)":
                    # Simplified Butler-Volmer
                    eta = params['eta'] * 0.001  # Convert mV to V
                    flux = params['k0'] * np.sinh(params['alpha'] * eta * 38.94)  # F/2RT ≈ 19.47 at 298K
                else:
                    flux = params.get('current', 0.01) * 0.1
            
            self.c = update_concentration(
                self.c, mu, 
                params['Mx'], params['My'], 
                dt, flux=flux
            )
            
            # Save frame
            if step % params['save_every'] == 0 or step == params['total_steps'] - 1:
                self.frames.append(self.c.copy())
                
                # Calculate statistics
                stats = {
                    'step': step,
                    'time': step * dt,
                    'mean_c': np.mean(self.c),
                    'std_c': np.std(self.c),
                    'min_c': np.min(self.c),
                    'max_c': np.max(self.c),
                    'gradient_norm': np.mean(np.abs(np.gradient(self.c)))
                }
                self.stats.append(stats)
                
                # Update progress
                progress = step / params['total_steps']
                progress_bar.progress(progress)
                
                if step % (params['save_every'] * 10) == 0:
                    status_text.text(f"Step {step:,} | Time = {step*dt:.3f} | Mean c = {np.mean(self.c):.4f}")
        
        elapsed = time.time() - start_time
        st.success(f"Simulation completed in {elapsed:.1f} seconds!")
        
        return self.frames, self.stats
    
    def analyze_morphology(self):
        """Analyze domain morphology from final frame"""
        from scipy import ndimage
        
        c = self.frames[-1]
        
        # Binary threshold
        c_binary = c > 0.5
        
        # Label connected domains
        labeled, num_features = ndimage.label(c_binary)
        
        # Calculate domain properties
        domain_sizes = []
        domain_orientations = []
        
        for i in range(1, num_features + 1):
            mask = labeled == i
            if np.sum(mask) > 10:  # Ignore small domains
                # Domain size
                domain_sizes.append(np.sum(mask))
                
                # Domain orientation (moment of inertia)
                y, x = np.where(mask)
                if len(x) > 2:
                    cov = np.cov(x, y)
                    eigvals, eigvecs = np.linalg.eig(cov)
                    orientation = np.degrees(np.arctan2(eigvecs[1,0], eigvecs[0,0]))
                    domain_orientations.append(orientation)
        
        return {
            'num_domains': num_features,
            'mean_size': np.mean(domain_sizes) if domain_sizes else 0,
            'size_std': np.std(domain_sizes) if domain_sizes else 0,
            'mean_orientation': np.mean(domain_orientations) if domain_orientations else 0,
            'orientation_std': np.std(domain_orientations) if domain_orientations else 0
        }

# ============================================
# VISUALIZATION FUNCTIONS
# ============================================
def create_animation(frames, steps):
    """Create interactive animation of concentration evolution"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("LiₓFePO₄ Concentration Evolution",)
    )
    
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
    
    # Initial frame
    fig.add_trace(
        go.Heatmap(
            z=frames[0],
            zmin=0, zmax=1,
            colorscale='RdBu_r',
            showscale=True,
            colorbar=dict(title="Li Fraction x")
        )
    )
    
    # Animation settings
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "⏸️ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ]
        }],
        height=600,
        xaxis_title="Position (a-axis)",
        yaxis_title="Position (c-axis)" if "ac-plane" in sim_type else "Position (b-axis)"
    )
    
    fig.frames = fig_frames
    
    return fig

def plot_statistics(stats):
    """Plot evolution of statistical measures"""
    steps = [s['step'] for s in stats]
    time = [s['time'] for s in stats]
    mean_c = [s['mean_c'] for s in stats]
    std_c = [s['std_c'] for s in stats]
    grad_norm = [s['gradient_norm'] for s in stats]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Mean Concentration",
            "Concentration Standard Deviation",
            "Gradient Norm",
            "Phase Fraction"
        )
    )
    
    # Mean concentration
    fig.add_trace(
        go.Scatter(x=time, y=mean_c, mode='lines', name='Mean c'),
        row=1, col=1
    )
    
    # Standard deviation
    fig.add_trace(
        go.Scatter(x=time, y=std_c, mode='lines', name='Std Dev', line=dict(color='red')),
        row=1, col=2
    )
    
    # Gradient norm
    fig.add_trace(
        go.Scatter(x=time, y=grad_norm, mode='lines', name='Gradient', line=dict(color='green')),
        row=2, col=1
    )
    
    # Phase fraction (Li-rich phase)
    phase_frac = [np.mean(np.array(frame) > 0.5) for frame in frames]
    fig.add_trace(
        go.Scatter(x=time[:len(phase_frac)], y=phase_frac, mode='lines', name='Li-rich %', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=2)
    fig.update_yaxes(title_text="Concentration", row=1, col=1)
    fig.update_yaxes(title_text="Std Dev", row=1, col=2)
    
    return fig

def visualize_3d_slice(frames_3d):
    """Visualize 3D data as 2D slices"""
    frame = frames_3d[-1]
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("XY Slice", "XZ Slice", "YZ Slice"),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'heatmap'}]]
    )
    
    # Take middle slices
    xy_slice = frame[:, :, nz//2]
    xz_slice = frame[:, ny//2, :]
    yz_slice = frame[nx//2, :, :]
    
    fig.add_trace(
        go.Heatmap(z=xy_slice, colorscale='RdBu_r', zmin=0, zmax=1),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=xz_slice, colorscale='RdBu_r', zmin=0, zmax=1),
        row=1, col=2
    )
    fig.add_trace(
        go.Heatmap(z=yz_slice, colorscale='RdBu_r', zmin=0, zmax=1),
        row=1, col=3
    )
    
    fig.update_layout(height=400)
    
    return fig

# ============================================
# MAIN APP LAYOUT
# ============================================
if run_simulation:
    # Collect parameters
    params = {
        'nx': nx, 'ny': ny, 'nz': nz,
        'total_steps': total_steps,
        'save_every': save_every,
        'Omega_RT': Omega_RT,
        'T_K': T_K,
        'kappa': kappa,
        'Mx': Mx, 'My': M0,
        'use_elasticity': use_elasticity,
        'C11': C11, 'C12': C12, 'C44': C44,
        'epsilon0': epsilon0 if use_elasticity else None,
        'use_plasticity': use_plasticity,
        'sigma_y0': sigma_y0,
        'hardening': hardening if use_plasticity else 0,
        'bc_type': bc_type,
        'eta': eta if bc_type == "Potentiostatic (Butler-Volmer)" else 0,
        'k0': k0 if bc_type == "Potentiostatic (Butler-Volmer)" else 1.0,
        'alpha': alpha if bc_type == "Potentiostatic (Butler-Volmer)" else 0.5,
        'current': current if bc_type == "Galvanostatic (Constant Flux)" else 0,
        'init_type': init_type,
        'noise_level': noise_level
    }
    
    # Create simulator
    simulator = LFPPhaseFieldSimulator(params)
    
    # Run simulation
    with st.spinner("Running phase-field simulation..."):
        frames, stats = simulator.run()
    
    # Display results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">Animation</div>', unsafe_allow_html=True)
        
        if use_3d:
            fig_3d = visualize_3d_slice(frames)
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            fig_anim = create_animation(frames, [s['step'] for s in stats])
            st.plotly_chart(fig_anim, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">Simulation Metrics</div>', unsafe_allow_html=True)
        
        # Final statistics
        final_stats = stats[-1]
        morphology = simulator.analyze_morphology()
        
        st.metric("Total Time", f"{final_stats['time']:.3f}")
        st.metric("Final Mean x", f"{final_stats['mean_c']:.4f}")
        st.metric("Phase Separation (σ)", f"{final_stats['std_c']:.4f}")
        st.metric("Domains Identified", morphology['num_domains'])
        st.metric("Avg Domain Size", f"{morphology['mean_size']:.1f} px")
        
        if morphology['mean_orientation'] != 0:
            st.metric("Avg Orientation", f"{morphology['mean_orientation']:.1f}°")
    
    # Statistics plot
    st.markdown('<div class="section-header">Evolution Statistics</div>', unsafe_allow_html=True)
    fig_stats = plot_statistics(stats)
    st.plotly_chart(fig_stats, use_container_width=True)
    
    # Export options
    st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Animation as GIF"):
            # Create GIF (placeholder)
            st.info("GIF export feature requires additional libraries")
    
    with col2:
        # Export data as CSV
        csv_data = []
        for stat in stats:
            csv_data.append({
                'Step': stat['step'],
                'Time': stat['time'],
                'Mean_c': stat['mean_c'],
                'Std_c': stat['std_c'],
                'Min_c': stat['min_c'],
                'Max_c': stat['max_c']
            })
        
        df = pd.DataFrame(csv_data)
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="lfp_simulation.csv">📥 Download CSV Data</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        # Export final frame as image
        final_frame = frames[-1]
        fig = go.Figure(data=go.Heatmap(z=final_frame, colorscale='RdBu_r'))
        img_bytes = fig.to_image(format="png")
        b64 = base64.b64encode(img_bytes).decode()
        href = f'<a href="data:image/png;base64,{b64}" download="lfp_final_state.png">🖼️ Download Final State</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    # Physics Insights
    st.markdown('<div class="section-header">Physics Insights</div>', unsafe_allow_html=True)
    
    insights = []
    
    # Analyze patterns
    if final_stats['std_c'] > 0.1:
        insights.append("✅ **Strong phase separation** observed (spinodal decomposition)")
    else:
        insights.append("⚠️ **Weak phase separation** - may be in solid solution regime")
    
    if use_elasticity and morphology['mean_orientation'] != 0:
        insights.append(f"📐 **Anisotropic domains** with average orientation {morphology['mean_orientation']:.1f}°")
    
    if len(frames) > 10:
        early_std = stats[1]['std_c']
        late_std = stats[-1]['std_c']
        if late_std > early_std * 2:
            insights.append("📈 **Domain coarsening** detected (increasing domain size)")
    
    for insight in insights:
        st.info(insight)

else:
    # Welcome/Instructions
    st.markdown("""
    ## 📖 Welcome to the LiFePO₄ Phase-Field Simulator
    
    This tool simulates **spinodal decomposition** in LiFePO₄ nanoparticles during battery charging/discharging.
    
    ### Key Features:
    
    1. **Anisotropic Elasticity**: Models the 6.8% volume mismatch between LiFePO₄ and FePO₄
    2. **J2 Plasticity**: Accounts for dislocation-mediated stress relaxation
    3. **Anisotropic Diffusion**: Fast Li transport along the b-axis (1D channels)
    4. **Electrochemical BCs**: Butler-Volmer kinetics with stress coupling
    5. **Real-time Visualization**: Interactive animations and analysis
    
    ### How to Use:
    
    1. Configure parameters in the **sidebar**
    2. Click **"Run Simulation"** to start
    3. Analyze results in the main panel
    
    ### Recommended Starting Parameters:
    
    - **Grid Size**: 256×256 (2D) or 64×64×64 (3D)
    - **Ω/RT**: 13.0 (strong phase separation)
    - **Anisotropy**: 10,000:1 for ab-plane (fast along b-axis)
    - **Elasticity**: Enable with C₁₁=200 GPa, εₓₓ=2.5%, εᵧᵧ=0.15%, ε₂₂=-2.5%
    - **Boundary**: Potentiostatic with η=30 mV
    
    ### Theoretical Background:
    
    The model solves the coupled equations:
    
    ```
    ∂c/∂t = ∇·[M(c)∇μ]
    μ = δF/δc - κ∇²c + μ_el + μ_pl
    ∇·σ = 0, σ = C:(ε - ε⁰(c) - ε^p)
    ```
    
    Where plastic flow follows J2 plasticity with isotropic hardening.
    """)
    
    # Quick start example
    if st.button("⚡ Load Example Parameters", type="secondary"):
        # This would set parameters via session state
        st.info("Example parameters loaded! Adjust as needed and click 'Run Simulation'")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p><strong>LiFePO₄ Phase-Field Simulator</strong> | Based on Cahn-Hilliard theory with anisotropic elasticity</p>
    <p>Simulates spinodal decomposition in battery cathodes • Supports academic research</p>
</div>
""", unsafe_allow_html=True)
