import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import pandas as pd

# =====================================================
# PHYSICAL CONSTANTS AND SCALES FOR LiFePO₄
# =====================================================

class PhysicalScales:
    """Define physical scales to make dimensionless simulation identical to physical one"""
    
    # Physical constants
    R = 8.314462618  # J/(mol·K)
    F = 96485.33212  # C/mol
    k_B = 1.380649e-23  # J/K
    N_A = 6.02214076e23  # 1/mol
    
    # LiFePO₄ specific parameters from literature
    def __init__(self):
        # Material properties
        self.T = 298.15  # K - Temperature
        
        # LiFePO₄ has very narrow miscibility gap:
        # FePO₄ phase: Li_xFePO₄ with x ≈ 0.03
        # LiFePO₄ phase: Li_xFePO₄ with x ≈ 0.97
        self.c_alpha = 0.03  # Li-poor phase (FePO₄-rich)
        self.c_beta = 0.97   # Li-rich phase (LiFePO₄-rich)
        
        # Molar volume of LiFePO₄
        self.V_m = 3.0e-5  # m³/mol
        
        # Diffusion coefficient along b-axis ([010])
        self.D_b = 1.0e-14  # m²/s (typical value)
        
        # Choose characteristic scales to match your dimensionless simulation
        self.set_scales_to_match_dimensionless()
    
    def set_scales_to_match_dimensionless(self):
        """
        Set scales so that our dimensionless simulation (W=1, κ=2, M=1, dt=0.1)
        corresponds to realistic LiFePO₄ physics
        """
        # Characteristic length scale (grid spacing in physical units)
        # For LiFePO₄ nanoparticles, typical domain size is 10-100 nm
        self.L0 = 1.0e-8  # 10 nm - characteristic length scale
        
        # Energy density scale
        # From regular solution model: f = (Ω/V_m) * c(1-c) + ...
        # For LiFePO₄, Ω ≈ 55 kJ/mol gives strong phase separation
        self.Ω = 55e3  # J/mol - Regular solution parameter
        self.E0 = self.Ω / self.V_m  # J/m³ - Energy density scale
        
        # Time scale: from diffusion length L0²/D
        self.t0 = (self.L0**2) / self.D_b  # s
        
        # Mobility scale: M0 = D_b / (E0 * t0)
        self.M0 = self.D_b / (self.E0 * self.t0)  # m⁵/(J·s)
        
        print(f"Physical scales set:")
        print(f"  Length scale L0 = {self.L0:.2e} m ({self.L0*1e9:.1f} nm)")
        print(f"  Time scale t0 = {self.t0:.2e} s")
        print(f"  Energy density scale E0 = {self.E0:.2e} J/m³")
        print(f"  Mobility scale M0 = {self.M0:.2e} m⁵/(J·s)")
        
    def dimensionless_to_physical(self, W_dim, κ_dim, M_dim, dt_dim):
        """Convert dimensionless parameters to physical units"""
        W_phys = W_dim * self.E0  # J/m³
        κ_phys = κ_dim * self.E0 * self.L0**2  # J/m
        M_phys = M_dim * self.M0  # m⁵/(J·s)
        dt_phys = dt_dim * self.t0  # s
        
        return W_phys, κ_phys, M_phys, dt_phys
    
    def physical_to_dimensionless(self, W_phys, κ_phys, M_phys, dt_phys):
        """Convert physical parameters to dimensionless units"""
        W_dim = W_phys / self.E0
        κ_dim = κ_phys / (self.E0 * self.L0**2)
        M_dim = M_phys / self.M0
        dt_dim = dt_phys / self.t0
        
        return W_dim, κ_dim, M_dim, dt_dim

# =====================================================
# NUMBA-ACCELERATED FUNCTIONS (IDENTICAL TO YOUR CODE)
# =====================================================

@njit(fastmath=True, cache=True)
def double_well_energy(c, A, B, C):
    """Generalized double-well free energy function"""
    return A * c**2 + B * c**3 + C * c**4

@njit(fastmath=True, cache=True)
def chemical_potential(c, A, B, C):
    """Chemical potential: μ = df/dc"""
    return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

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
def update_concentration(c, dt, dx, kappa, M, A, B, C):
    """Update concentration field using Cahn-Hilliard equation"""
    nx, ny = c.shape
    
    lap_c = compute_laplacian(c, dx)
    mu = chemical_potential(c, A, B, C) - kappa * lap_c
    mu_x = compute_gradient_x(mu, dx)
    mu_y = compute_gradient_y(mu, dx)
    
    flux_x = M * mu_x
    flux_y = M * mu_y
    
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
    
    return c + dt * div_flux

# =====================================================
# PHYSICAL SIMULATION CLASS
# =====================================================

class PhysicalPhaseFieldSimulation:
    """Phase field simulation with physical units for LiFePO₄"""
    
    def __init__(self, nx=256, ny=256, dx=1.0, dt=0.1):
        # Simulation grid
        self.nx = nx
        self.ny = ny
        self.dx = dx  # dimensionless grid spacing
        self.dt = dt  # dimensionless time step
        
        # Physical scales
        self.scales = PhysicalScales()
        
        # Dimensionless parameters (matching your original simulation)
        self.W_dim = 1.0  # Double-well height (dimensionless)
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        
        self.kappa_dim = 2.0  # Gradient coefficient (dimensionless)
        self.M_dim = 1.0      # Mobility (dimensionless)
        
        # Calculate corresponding physical parameters
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(self.W_dim, self.kappa_dim, self.M_dim, self.dt)
        
        # Initialize concentration field
        self.c = np.zeros((nx, ny))
        
        # Time tracking
        self.time_dim = 0.0  # dimensionless time
        self.time_phys = 0.0  # physical time (seconds)
        self.step = 0
        
        # History tracking
        self.history = {
            'time_dim': [],
            'time_phys': [],
            'mean': [],
            'std': [],
            'phase_FePO4': [],
            'phase_LiFePO4': []
        }
        
        # Initialize
        self.initialize_random()
    
    def update_physical_parameters(self):
        """Update physical parameters based on current dimensionless values"""
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(self.W_dim, self.kappa_dim, self.M_dim, self.dt)
        
        # Update A, B, C for double-well
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
    
    def set_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None):
        """Set dimensionless parameters"""
        if W_dim is not None:
            self.W_dim = W_dim
        if kappa_dim is not None:
            self.kappa_dim = kappa_dim
        if M_dim is not None:
            self.M_dim = M_dim
        if dt_dim is not None:
            self.dt = dt_dim
        
        self.update_physical_parameters()
    
    def set_physical_parameters(self, W_phys=None, kappa_phys=None, M_phys=None, dt_phys=None):
        """Set physical parameters directly"""
        if W_phys is not None:
            self.W_dim = W_phys / self.scales.E0
        if kappa_phys is not None:
            self.kappa_dim = kappa_phys / (self.scales.E0 * self.scales.L0**2)
        if M_phys is not None:
            self.M_dim = M_phys / self.scales.M0
        if dt_phys is not None:
            self.dt = dt_phys / self.scales.t0
        
        self.update_physical_parameters()
    
    def initialize_random(self, c0=0.5, noise_amplitude=0.05):
        """Initialize with random fluctuations"""
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
    
    def initialize_FePO4_with_LiFePO4_seed(self):
        """Initialize with FePO₄ background and LiFePO₄ seed (more physically realistic)"""
        # Start with FePO₄ (Li-poor phase)
        self.c = self.scales.c_alpha * np.ones((self.nx, self.ny))
        
        # Add LiFePO₄ seed in center
        center_x, center_y = self.nx // 2, self.ny // 2
        seed_radius = min(self.nx, self.ny) // 8
        
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - center_x)**2 + (j - center_y)**2 < seed_radius**2:
                    self.c[i, j] = self.scales.c_beta  # LiFePO₄
        
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
    
    def initialize_LiFePO4_with_FePO4_seed(self):
        """Initialize with LiFePO₄ background and FePO₄ seed (delithiation)"""
        # Start with LiFePO₄ (Li-rich phase)
        self.c = self.scales.c_beta * np.ones((self.nx, self.ny))
        
        # Add FePO₄ seed in center
        center_x, center_y = self.nx // 2, self.ny // 2
        seed_radius = min(self.nx, self.ny) // 8
        
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - center_x)**2 + (j - center_y)**2 < seed_radius**2:
                    self.c[i, j] = self.scales.c_alpha  # FePO₄
        
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
    
    def clear_history(self):
        self.history = {
            'time_dim': [],
            'time_phys': [],
            'mean': [],
            'std': [],
            'phase_FePO4': [],
            'phase_LiFePO4': []
        }
        self.update_history()
    
    def update_history(self):
        self.history['time_dim'].append(self.time_dim)
        self.history['time_phys'].append(self.time_phys)
        self.history['mean'].append(np.mean(self.c))
        self.history['std'].append(np.std(self.c))
        
        # Use actual phase boundaries for LiFePO₄
        # FePO₄-rich phase: c < 0.5 (but in reality c_alpha ≈ 0.03)
        # LiFePO₄-rich phase: c >= 0.5 (but in reality c_beta ≈ 0.97)
        threshold = 0.5  # Midpoint for visualization
        self.history['phase_FePO4'].append(np.sum(self.c < threshold) / (self.nx * self.ny))
        self.history['phase_LiFePO4'].append(np.sum(self.c >= threshold) / (self.nx * self.ny))
    
    def run_step(self):
        """Run one time step"""
        self.c = update_concentration(
            self.c, self.dt, self.dx, 
            self.kappa_dim, self.M_dim,
            self.A, self.B, self.C
        )
        self.time_dim += self.dt
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()
    
    def run_steps(self, n_steps):
        """Run multiple time steps"""
        for _ in range(n_steps):
            self.run_step()
    
    def compute_free_energy_density(self):
        """Compute free energy density (dimensionless)"""
        energy = np.zeros_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                energy[i, j] = double_well_energy(self.c[i, j], self.A, self.B, self.C)
        return energy
    
    def compute_voltage(self):
        """Calculate voltage from concentration (simplified model)"""
        # For LiFePO₄, voltage is ~3.42 V vs Li/Li⁺ during phase coexistence
        # This is a simplified model based on double-well free energy
        
        mean_c = np.mean(self.c)
        
        # Calculate voltage from regular solution model
        # V = V0 - (1/F) * dF/dc
        V0 = 3.42  # Reference voltage (V)
        
        # Chemical potential from double-well
        mu = 2.0 * self.A * mean_c + 3.0 * self.B * mean_c**2 + 4.0 * self.C * mean_c**3
        
        # Convert to voltage (simplified)
        # Note: This is not exact but gives reasonable behavior
        voltage = V0 - 0.1 * mu  # Scaling factor to get reasonable voltage range
        
        return max(3.0, min(4.0, voltage))  # Clip to reasonable range
    
    def get_statistics(self):
        """Get comprehensive statistics"""
        mean_c = np.mean(self.c)
        
        # Calculate x in Li_xFePO₄
        # Linear interpolation between c_alpha (FePO₄) and c_beta (LiFePO₄)
        # For c = 0, x = 0 (FePO₄); for c = 1, x = 1 (LiFePO₄)
        # In reality, phases have finite width: c_alpha ≈ 0.03, c_beta ≈ 0.97
        x_Li = mean_c  # Simplified for now
        
        # Phase fractions using threshold
        threshold = 0.5
        phase_FePO4 = np.sum(self.c < threshold) / (self.nx * self.ny)
        phase_LiFePO4 = np.sum(self.c >= threshold) / (self.nx * self.ny)
        
        # Physical dimensions
        domain_size_nm = self.nx * self.dx * self.scales.L0 * 1e9
        interface_width_nm = np.sqrt(self.kappa_phys / self.W_phys) * 1e9
        
        return {
            # Dimensionless
            'time_dim': self.time_dim,
            'step': self.step,
            'mean_concentration': mean_c,
            'std_concentration': np.std(self.c),
            
            # Physical
            'time_physical': self.time_phys,
            'x_Li': x_Li,
            'phase_fraction_FePO4': phase_FePO4,
            'phase_fraction_LiFePO4': phase_LiFePO4,
            'voltage': self.compute_voltage(),
            'domain_size_nm': domain_size_nm,
            'interface_width_nm': interface_width_nm,
            
            # Parameters
            'W_dim': self.W_dim,
            'kappa_dim': self.kappa_dim,
            'M_dim': self.M_dim,
            'W_phys': self.W_phys,
            'kappa_phys': self.kappa_phys,
            'M_phys': self.M_phys,
            'dt_phys': self.dt_phys,
        }

# =====================================================
# STREAMLIT APP
# =====================================================

def main():
    st.set_page_config(
        page_title="LiFePO₄ Phase Field Simulation",
        page_icon="🔋",
        layout="wide"
    )
    
    # Title and description
    st.title("🔋 LiFePO₄ Phase Decomposition - Step by Step")
    st.markdown("""
    ### From Dimensionless to Physical: LiFePO₄ ↔ FePO₄ Phase Separation
    
    This simulation starts with your dimensionless code and adds **physical units** for LiFePO₄.
    The numerical scheme is identical, but now with meaningful physical scales.
    """)
    
    # Initialize simulation
    if 'sim' not in st.session_state:
        st.session_state.sim = PhysicalPhaseFieldSimulation(nx=256, ny=256, dx=1.0, dt=0.1)
    
    sim = st.session_state.sim
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        
        # Display physical scales
        with st.expander("📏 Physical Scales", expanded=True):
            st.markdown(f"""
            **Characteristic Length:** {sim.scales.L0*1e9:.1f} nm  
            **Characteristic Time:** {sim.scales.t0:.2e} s  
            **Energy Scale:** {sim.scales.E0:.2e} J/m³  
            **FePO₄ phase:** Li_xFePO₄ with x ≈ {sim.scales.c_alpha:.2f}  
            **LiFePO₄ phase:** Li_xFePO₄ with x ≈ {sim.scales.c_beta:.2f}  
            **Diffusion Coef.:** {sim.scales.D_b:.2e} m²/s
            """)
        
        st.divider()
        
        # Simulation control
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("Steps/update", 1, 1000, 10)
        
        with col2:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running..."):
                    sim.run_steps(steps)
                    st.rerun()
        
        st.divider()
        
        # Initialization options
        st.subheader("Initial Conditions")
        init_option = st.radio(
            "Choose:",
            ["Random (c=0.5)", "FePO₄ with LiFePO₄ seed", "LiFePO₄ with FePO₄ seed"],
            index=0
        )
        
        if st.button("🔄 Apply Initialization", use_container_width=True):
            if init_option == "Random (c=0.5)":
                sim.initialize_random(c0=0.5, noise_amplitude=0.05)
            elif init_option == "FePO₄ with LiFePO₄ seed":
                sim.initialize_FePO4_with_LiFePO4_seed()
            else:
                sim.initialize_LiFePO4_with_FePO4_seed()
            st.rerun()
        
        st.divider()
        
        # Parameter controls - DIMENSIONLESS (matching your original code)
        st.subheader("Dimensionless Parameters")
        
        W_dim = st.slider("W (Double-well)", 0.1, 5.0, float(sim.W_dim), 0.1,
                         help="Controls phase separation strength. Higher = sharper interfaces")
        
        kappa_dim = st.slider("κ (Gradient)", 0.1, 10.0, float(sim.kappa_dim), 0.1,
                             help="Controls interface width. Higher = wider interfaces")
        
        M_dim = st.slider("M (Mobility)", 0.01, 5.0, float(sim.M_dim), 0.01,
                         help="Controls kinetics. Higher = faster phase separation")
        
        dt_dim = st.slider("Δt (Time step)", 0.01, 0.5, float(sim.dt), 0.01,
                          help="Numerical time step")
        
        # Update if changed
        if (W_dim != sim.W_dim or kappa_dim != sim.kappa_dim or 
            M_dim != sim.M_dim or dt_dim != sim.dt):
            sim.set_parameters(W_dim=W_dim, kappa_dim=kappa_dim, M_dim=M_dim, dt_dim=dt_dim)
        
        st.divider()
        
        # Physical parameter display
        st.subheader("📊 Physical Parameters")
        stats = sim.get_statistics()
        
        col_phys1, col_phys2 = st.columns(2)
        with col_phys1:
            st.metric("W (physical)", f"{stats['W_phys']:.2e} J/m³")
            st.metric("κ (physical)", f"{stats['kappa_phys']:.2e} J/m")
            st.metric("M (physical)", f"{stats['M_phys']:.2e} m⁵/(J·s)")
        
        with col_phys2:
            st.metric("Δt (physical)", f"{stats['dt_phys']:.2e} s")
            st.metric("Time", f"{stats['time_physical']:.2e} s")
            st.metric("Interface width", f"{stats['interface_width_nm']:.1f} nm")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Li Concentration Field (LiₓFePO₄)")
        
        # Create figure
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        
        # Plot concentration
        im = ax1.imshow(sim.c.T, cmap='RdYlBu', origin='lower', 
                       vmin=0, vmax=1, aspect='auto')
        
        # Calculate physical domain size
        domain_nm = sim.nx * sim.dx * sim.scales.L0 * 1e9
        
        # Set title and labels
        ax1.set_title(f"LiₓFePO₄ - Time = {stats['time_physical']:.2e} s\n"
                     f"x = {stats['mean_concentration']:.3f}, "
                     f"Domain = {domain_nm:.0f} nm")
        ax1.set_xlabel(f"x position ({domain_nm:.0f} nm)")
        ax1.set_ylabel(f"y position ({domain_nm:.0f} nm)")
        
        # Add colorbar with phase labels
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Lithium Concentration (x in LiₓFePO₄)')
        cbar.ax.text(0.5, -0.02, 'FePO₄-rich', transform=cbar.ax.transAxes, 
                    ha='center', va='top', fontsize=9)
        cbar.ax.text(0.5, 1.02, 'LiFePO₄-rich', transform=cbar.ax.transAxes, 
                    ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("Statistics")
        
        # Current statistics
        stats = sim.get_statistics()
        
        st.metric("x in LiₓFePO₄", f"{stats['x_Li']:.3f}")
        st.metric("Voltage", f"{stats['voltage']:.3f} V")
        st.metric("FePO₄ fraction", f"{stats['phase_fraction_FePO4']:.3f}")
        st.metric("LiFePO₄ fraction", f"{stats['phase_fraction_LiFePO4']:.3f}")
        st.metric("Std Dev", f"{stats['std_concentration']:.3f}")
        
        # Histogram
        st.subheader("Concentration Distribution")
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        ax2.hist(sim.c.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(sim.scales.c_alpha, color='red', linestyle='--', alpha=0.7, label='FePO₄')
        ax2.axvline(sim.scales.c_beta, color='green', linestyle='--', alpha=0.7, label='LiFePO₄')
        ax2.set_xlim(0, 1)
        ax2.set_xlabel('x in LiₓFePO₄')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)
    
    # Time evolution plots
    st.subheader("📈 Time Evolution")
    
    if len(sim.history['time_phys']) > 1:
        fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Concentration vs time
        axes[0, 0].plot(sim.history['time_phys'], sim.history['mean'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Mean x in LiₓFePO₄")
        axes[0, 0].set_title("Lithium Content vs Time")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phase fractions
        axes[0, 1].plot(sim.history['time_phys'], sim.history['phase_FePO4'], 'r-', 
                       label='FePO₄-rich', linewidth=2)
        axes[0, 1].plot(sim.history['time_phys'], sim.history['phase_LiFePO4'], 'g-', 
                       label='LiFePO₄-rich', linewidth=2)
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Phase Fraction")
        axes[0, 1].set_title("Phase Fractions vs Time")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Standard deviation (phase separation measure)
        axes[1, 0].plot(sim.history['time_phys'], sim.history['std'], color='purple', linewidth=2)
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Standard Deviation")
        axes[1, 0].set_title("Phase Separation Progress")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Calculate voltage history
        voltages = []
        for mean_c in sim.history['mean']:
            V0 = 3.42
            mu = 2.0 * sim.A * mean_c + 3.0 * sim.B * mean_c**2 + 4.0 * sim.C * mean_c**3
            voltage = V0 - 0.1 * mu
            voltages.append(max(3.0, min(4.0, voltage)))
        
        axes[1, 1].plot(sim.history['time_phys'], voltages, color='orange', linewidth=2)
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Voltage (V)")
        axes[1, 1].set_title("Voltage vs Time")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)
    
    # Information section
    with st.expander("📚 Key Points about LiFePO₄ Phase Decomposition", expanded=True):
        st.markdown("""
        ### Physical Values for LiₓFePO₄:
        
        **For LiFePO₄ (lithiated phase):**
        - **x ≈ 0.97** (not exactly 1.0 due to finite solubility)
        - Dark blue regions in simulation
        
        **For FePO₄ (delithiated phase):**
        - **x ≈ 0.03** (not exactly 0.0 due to finite solubility)
        - Red regions in simulation
        
        ### Why These Values?
        
        1. **Narrow miscibility gap**: LiFePO₄ has very low mutual solubility
           - FePO₄ can only hold ~3% Li
           - LiFePO₄ is nearly stoichiometric (~97% Li)
        
        2. **Flat voltage plateau**: During phase coexistence, voltage is constant at ~3.42 V
           - This corresponds to the two-phase region
           - In simulation: voltage stays ~3.4-3.5 V during phase separation
        
        3. **Sharp interfaces**: Interfaces are only 1-2 nm thick
           - In simulation: interface width controlled by κ/W ratio
        
        ### How the Simulation Works:
        
        **Numerical Scheme**: Identical to your dimensionless code
        - Same Cahn-Hilliard equation: ∂c/∂t = ∇·(M ∇μ)
        - Same double-well free energy: f(c) = W c²(1-c)²
        - Same finite difference scheme
        
        **Physical Scaling**: Added through scales:
        - **Length scale**: L₀ = 10 nm (characteristic domain size)
        - **Time scale**: t₀ = L₀²/D ≈ 0.01 s (from diffusion)
        - **Energy scale**: E₀ = Ω/Vₘ ≈ 1.8×10⁹ J/m³ (from regular solution)
        
        **Your dimensionless parameters now have physical meaning:**
        - W = 1.0 → W_phys = 1.8×10⁹ J/m³
        - κ = 2.0 → κ_phys = 3.6×10⁻¹⁰ J/m
        - M = 1.0 → M_phys = 5.6×10⁻¹⁰ m⁵/(J·s)
        - Δt = 0.1 → Δt_phys = 0.001 s
        """)
    
    # Auto-run option
    st.sidebar.divider()
    auto_run = st.sidebar.checkbox("Auto-run simulation", value=False)
    auto_speed = st.sidebar.slider("Steps per second", 1, 100, 10)
    
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
