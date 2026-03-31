import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import time

# =====================================================
# Physical Scales Class for LiFePO₄
# =====================================================

class PhysicalScalesLiFePO4:
    """
    Physical unit conversion for LiFePO₄ phase-field simulations.
    All scales based on literature values for LiFePO₄ cathode materials.
    """
    def __init__(self, L0_nm=10.0, D_b=1e-14, Omega_kJmol=12.0, T=298.15):
        # Fundamental constants
        self.R = 8.314462618          # J/(mol·K)
        self.T = T                    # K
        self.V_m = 4.38e-5            # m³/mol (molar volume LiFePO₄)
        
        # Material parameters (user-configurable via sliders later)
        self.D_b = D_b                # m²/s (diffusion coefficient)
        self.Omega = Omega_kJmol * 1e3  # J/mol (regular solution parameter)
        
        # Characteristic scales
        self.L0 = L0_nm * 1e-9        # m (reference length from slider)
        self.E0 = self.Omega / self.V_m   # J/m³ (energy density scale)
        self.t0 = self.L0**2 / self.D_b   # s (diffusion time scale)
        
        # Mobility scale: M0 = D_b * V_m / (R*T) ≈ D_b / (E0) for regular solution
        self.M0 = self.D_b / self.E0      # m⁵/(J·s)
        
    def dim_to_phys(self, W_dim, kappa_dim, M_dim, dt_dim, dx_dim=1.0):
        """Convert dimensionless parameters to physical SI units"""
        W_phys = W_dim * self.E0                      # J/m³
        kappa_phys = kappa_dim * self.E0 * self.L0**2 # J/m
        M_phys = M_dim * self.M0                       # m⁵/(J·s)
        dt_phys = dt_dim * self.t0                     # s
        dx_phys = dx_dim * self.L0                     # m
        return W_phys, kappa_phys, M_phys, dt_phys, dx_phys
    
    def phys_to_interface_width(self, kappa_phys, W_phys):
        """Estimate interface width: ξ ≈ √(κ/W)"""
        return np.sqrt(kappa_phys / W_phys)  # meters
    
    def format_time(self, t_seconds):
        """Format physical time with appropriate units"""
        if t_seconds < 1e-6:
            return f"{t_seconds*1e9:.2f} ns"
        elif t_seconds < 1e-3:
            return f"{t_seconds*1e6:.2f} μs"
        elif t_seconds < 1.0:
            return f"{t_seconds*1e3:.2f} ms"
        elif t_seconds < 3600:
            return f"{t_seconds:.2f} s"
        else:
            return f"{t_seconds/3600:.2f} h"
    
    def format_length(self, L_meters):
        """Format length with appropriate units"""
        if L_meters < 1e-9:
            return f"{L_meters*1e10:.2f} Å"
        elif L_meters < 1e-6:
            return f"{L_meters*1e9:.2f} nm"
        elif L_meters < 1e-3:
            return f"{L_meters*1e6:.2f} μm"
        else:
            return f"{L_meters*1e3:.2f} mm"


# =====================================================
# Numba-accelerated Phase Field Functions (Physical Units)
# =====================================================

@njit(fastmath=True, cache=True)
def double_well_energy(c, A, B, C):
    """Generalized double-well free energy: f(c) = A·c² + B·c³ + C·c⁴"""
    return A * c**2 + B * c**3 + C * c**4

@njit(fastmath=True, cache=True)
def chemical_potential(c, A, B, C):
    """Chemical potential: μ = ∂f/∂c"""
    return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

@njit(fastmath=True, parallel=True)
def compute_laplacian(field, dx):
    """5-point stencil Laplacian with periodic BCs"""
    nx, ny = field.shape
    lap = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            im1, ip1 = (i - 1) % nx, (i + 1) % nx
            jm1, jp1 = (j - 1) % ny, (j + 1) % ny
            lap[i, j] = (field[ip1, j] + field[im1, j] + 
                         field[i, jp1] + field[i, jm1] - 
                         4.0 * field[i, j]) / (dx * dx)
    return lap

@njit(fastmath=True, parallel=True)
def compute_gradient_x(field, dx):
    """Central difference x-gradient with periodic BCs"""
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            ip1, im1 = (i + 1) % nx, (i - 1) % nx
            grad_x[i, j] = (field[ip1, j] - field[im1, j]) / (2.0 * dx)
    return grad_x

@njit(fastmath=True, parallel=True)
def compute_gradient_y(field, dx):
    """Central difference y-gradient with periodic BCs"""
    nx, ny = field.shape
    grad_y = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            jp1, jm1 = (j + 1) % ny, (j - 1) % ny
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dx)
    return grad_y

@njit(fastmath=True, parallel=True)
def update_concentration_physical(c, dt, dx, kappa, M, A, B, C):
    """
    Cahn-Hilliard update in physical units:
    ∂c/∂t = ∇·[M ∇(∂f/∂c - κ∇²c)]
    
    Parameters in SI units:
    - dt: seconds
    - dx: meters  
    - kappa: J/m (gradient energy coefficient)
    - M: m⁵/(J·s) (mobility)
    - A, B, C: J/m³ (free energy coefficients)
    """
    nx, ny = c.shape
    
    # Compute chemical potential μ = ∂f/∂c - κ∇²c
    lap_c = compute_laplacian(c, dx)
    mu_local = chemical_potential(c, A, B, C)
    mu = mu_local - kappa * lap_c
    
    # Compute flux J = -M ∇μ
    mu_x = compute_gradient_x(mu, dx)
    mu_y = compute_gradient_y(mu, dx)
    flux_x = -M * mu_x  # Note: negative sign for downhill diffusion
    flux_y = -M * mu_y
    
    # Compute divergence of flux
    div_flux = np.zeros_like(c)
    for i in prange(nx):
        for j in prange(ny):
            ip1, im1 = (i + 1) % nx, (i - 1) % nx
            jp1, jm1 = (j + 1) % ny, (j - 1) % ny
            div_x = (flux_x[ip1, j] - flux_x[im1, j]) / (2.0 * dx)
            div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dx)
            div_flux[i, j] = div_x + div_y
    
    return c + dt * div_flux


# =====================================================
# PhaseFieldSimulation Class (Physical Units)
# =====================================================

class PhaseFieldSimulation:
    def __init__(self, nx=256, ny=256, dx_dim=1.0, dt_dim=0.01, 
                 L0_nm=10.0, D_b=1e-14, Omega_kJmol=12.0):
        # Grid (kept dimensionless internally for stability)
        self.nx = nx
        self.ny = ny
        self.dx_dim = dx_dim  # dimensionless grid spacing
        
        # Physical scales
        self.scales = PhysicalScalesLiFePO4(L0_nm=L0_nm, D_b=D_b, Omega_kJmol=Omega_kJmol)
        
        # Dimensionless parameters (for numerical stability)
        self.W_dim = 1.0
        self.kappa_dim = 2.0
        self.M_dim = 1.0
        self.dt_dim = dt_dim
        
        # Physical parameters (computed from dimensionless)
        self._update_physical_params()
        
        # Free energy coefficients (dimensionless, converted to physical in update)
        self.A_dim = self.W_dim
        self.B_dim = -2.0 * self.W_dim
        self.C_dim = self.W_dim
        
        # State
        self.c = np.zeros((nx, ny))
        self.time_dim = 0.0  # dimensionless time
        self.step = 0
        self.history = {
            'time_dim': [], 'time_phys': [],
            'mean': [], 'std': [],
            'phase_high': [], 'phase_low': []
        }
        
    def _update_physical_params(self):
        """Convert dimensionless parameters to physical units"""
        (self.W_phys, self.kappa_phys, self.M_phys, 
         self.dt_phys, self.dx_phys) = self.scales.dim_to_phys(
            self.W_dim, self.kappa_dim, self.M_dim, self.dt_dim, self.dx_dim
        )
        # Convert free energy coefficients to physical
        self.A_phys = self.A_dim * self.scales.E0
        self.B_phys = self.B_dim * self.scales.E0
        self.C_phys = self.C_dim * self.scales.E0
        
    def set_physical_parameters(self, W_Jm3=None, kappa_Jm=None, M_m5Js=None, dt_s=None,
                                L0_nm=None, D_b=None, Omega_kJmol=None):
        """Set physical parameters directly (converts back to dimensionless internally)"""
        # Update scales if fundamental parameters changed
        if L0_nm is not None or D_b is not None or Omega_kJmol is not None:
            old_L0 = self.scales.L0
            self.scales = PhysicalScalesLiFePO4(
                L0_nm=L0_nm if L0_nm is not None else self.scales.L0*1e9,
                D_b=D_b if D_b is not None else self.scales.D_b,
                Omega_kJmol=Omega_kJmol if Omega_kJmol is not None else self.scales.Omega/1e3
            )
            # Rescale dimensionless params if L0 changed
            if L0_nm is not None and L0_nm != old_L0*1e9:
                ratio = (self.scales.L0 / old_L0)**2
                self.kappa_dim = self.kappa_dim / ratio  # κ scales with L0²
        
        # Convert physical → dimensionless
        if W_Jm3 is not None:
            self.W_dim = W_Jm3 / self.scales.E0
            self.A_dim = self.W_dim
            self.B_dim = -2.0 * self.W_dim
            self.C_dim = self.W_dim
        if kappa_Jm is not None:
            self.kappa_dim = kappa_Jm / (self.scales.E0 * self.scales.L0**2)
        if M_m5Js is not None:
            self.M_dim = M_m5Js / self.scales.M0
        if dt_s is not None:
            self.dt_dim = dt_s / self.scales.t0
            
        self._update_physical_params()
    
    def set_dimensionless_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None):
        """Set dimensionless parameters directly (for advanced users)"""
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
    
    def initialize_random(self, c0=0.5, noise_amplitude=0.01):
        """Initialize with random fluctuations (spinodal decomposition)"""
        np.random.seed(42)  # Reproducibility
        self.c = np.clip(c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0), 0, 1)
        self.time_dim = 0.0
        self.step = 0
        self.clear_history()
        
    def initialize_seed(self, c0=0.3, seed_value=0.7, radius_grid=15):
        """Initialize with circular seed (nucleation & growth)"""
        self.c = c0 * np.ones((self.nx, self.ny))
        center_x, center_y = self.nx // 2, self.ny // 2
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - center_x)**2 + (j - center_y)**2 < radius_grid**2:
                    self.c[i, j] = seed_value
        self.time_dim = 0.0
        self.step = 0
        self.clear_history()
    
    def clear_history(self):
        self.history = {
            'time_dim': [], 'time_phys': [],
            'mean': [], 'std': [],
            'phase_high': [], 'phase_low': []
        }
        self.update_history()
    
    def update_history(self):
        self.history['time_dim'].append(self.time_dim)
        self.history['time_phys'].append(self.time_dim * self.scales.t0)
        self.history['mean'].append(np.mean(self.c))
        self.history['std'].append(np.std(self.c))
        self.history['phase_high'].append(np.sum(self.c > 0.5) / (self.nx * self.ny))
        self.history['phase_low'].append(np.sum(self.c < 0.5) / (self.nx * self.ny))
    
    def run_step(self):
        """Run one time step using physical parameters"""
        self.c = update_concentration_physical(
            self.c, self.dt_phys, self.dx_phys,
            self.kappa_phys, self.M_phys,
            self.A_phys, self.B_phys, self.C_phys
        )
        # Clip concentration to [0, 1] for numerical stability
        self.c = np.clip(self.c, 0.0, 1.0)
        
        self.time_dim += self.dt_dim
        self.step += 1
        self.update_history()
    
    def run_steps(self, n_steps):
        for _ in range(n_steps):
            self.run_step()
    
    def compute_free_energy_density(self):
        """Return free energy density in J/m³"""
        energy = np.zeros_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                energy[i, j] = double_well_energy(
                    self.c[i, j], self.A_phys, self.B_phys, self.C_phys
                )
        return energy
    
    def get_statistics(self):
        """Return simulation statistics with physical units"""
        domain_size_m = self.nx * self.dx_phys
        interface_width_m = self.scales.phys_to_interface_width(self.kappa_phys, self.W_phys)
        diffusion_length_m = np.sqrt(self.scales.D_b * self.time_dim * self.scales.t0)
        
        return {
            'time_dim': self.time_dim,
            'time_phys': self.time_dim * self.scales.t0,
            'time_formatted': self.scales.format_time(self.time_dim * self.scales.t0),
            'step': self.step,
            'domain_size': self.scales.format_length(domain_size_m),
            'interface_width': self.scales.format_length(interface_width_m),
            'diffusion_length': self.scales.format_length(diffusion_length_m),
            'mean_concentration': np.mean(self.c),
            'std_concentration': np.std(self.c),
            'min_concentration': np.min(self.c),
            'max_concentration': np.max(self.c),
            'phase_fraction_high': np.sum(self.c > 0.5) / (self.nx * self.ny),
            'phase_fraction_low': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'W_phys': self.W_phys,
            'kappa_phys': self.kappa_phys,
            'M_phys': self.M_phys,
        }


# =====================================================
# Streamlit App (Physical Units Interface)
# =====================================================

def main():
    st.set_page_config(
        page_title="LiFePO₄ Phase-Field Simulation",
        page_icon="🔋",
        layout="wide"
    )
    
    st.title("🔋 LiₓFePO₄ Phase-Field Simulation: Spinodal Decomposition")
    st.markdown("""
    **Physically realistic 2D simulation** of phase separation in LiFePO₄ cathode particles.
    All parameters in **real units** (nm, s, J/m³). Based on the Cahn-Hilliard equation.
    """)
    
    # Initialize simulation
    if 'sim' not in st.session_state:
        st.session_state.sim = PhaseFieldSimulation(
            nx=256, ny=256, dx_dim=1.0, dt_dim=0.01,
            L0_nm=10.0, D_b=1e-14, Omega_kJmol=12.0
        )
        st.session_state.sim.initialize_random(c0=0.5, noise_amplitude=0.05)
    
    sim = st.session_state.sim
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Run controls
        st.subheader("⏱️ Time Stepping")
        steps_to_run = st.number_input("Steps per update", 1, 1000, 10)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run", use_container_width=True):
                with st.spinner("Computing..."):
                    sim.run_steps(steps_to_run)
        with col2:
            if st.button("⏹️ Pause", use_container_width=True):
                st.rerun()
        
        if st.button("🔄 Reset: Spinodal", use_container_width=True):
            sim.initialize_random(c0=0.5, noise_amplitude=0.05)
            st.rerun()
        if st.button("🌱 Reset: Nucleation", use_container_width=True):
            sim.initialize_seed(c0=0.3, seed_value=0.7, radius_grid=15)
            st.rerun()
        
        st.divider()
        
        # Material parameters
        st.subheader("🧪 Material Properties")
        
        L0_nm = st.slider("Reference length L₀ (nm)", 5.0, 100.0, 10.0, 1.0,
                         help="Characteristic length scale (interface width ~2-10 nm)")
        D_b_exp = st.slider("log₁₀(D_b) [m²/s]", -16, -10, -14, 1,
                           help="Diffusion coefficient along b-axis")
        D_b = 10**D_b_exp
        Omega_kJmol = st.slider("Ω (kJ/mol)", 5.0, 60.0, 12.0, 1.0,
                               help="Regular solution parameter (mixing enthalpy)")
        
        if st.button("Apply Material Parameters", use_container_width=True):
            sim.set_physical_parameters(L0_nm=L0_nm, D_b=D_b, Omega_kJmol=Omega_kJmol)
            st.rerun()
        
        st.divider()
        
        # Model parameters (physical units)
        st.subheader("⚙️ Model Parameters")
        
        # Compute reasonable slider ranges based on current scales
        W_range = sim.scales.E0 * np.array([0.1, 10.0])  # J/m³
        W_default = sim.W_phys
        
        kappa_range = sim.scales.E0 * sim.scales.L0**2 * np.array([0.1, 10.0])  # J/m
        kappa_default = sim.kappa_phys
        
        M_range = sim.scales.M0 * np.array([0.1, 10.0])  # m⁵/(J·s)
        M_default = sim.M_phys
        
        # Time step: ensure numerical stability (CFL-like condition for 4th order PDE)
        dt_max = 0.1 * sim.scales.t0  # Conservative stability limit
        dt_default = min(sim.dt_phys, dt_max)
        
        W_phys = st.number_input("W (J/m³)", 
                                float(W_range[0]), float(W_range[1]), 
                                float(W_default), format="%.2e",
                                help="Double-well barrier height")
        kappa_phys = st.number_input("κ (J/m)", 
                                    float(kappa_range[0]), float(kappa_range[1]), 
                                    float(kappa_default), format="%.2e",
                                    help="Gradient energy coefficient")
        M_phys = st.number_input("M (m⁵/J·s)", 
                                float(M_range[0]), float(M_range[1]), 
                                float(M_default), format="%.2e",
                                help="Mobility (kinetic coefficient)")
        dt_phys = st.number_input("Δt (s)", 
                                 1e-12, float(dt_max), 
                                 float(dt_default), format="%.2e",
                                 help="Time step (keep small for stability!)")
        
        if st.button("Apply Model Parameters", use_container_width=True):
            sim.set_physical_parameters(
                W_Jm3=W_phys, kappa_Jm=kappa_phys, 
                M_m5Js=M_phys, dt_s=dt_phys
            )
            st.rerun()
        
        st.divider()
        
        # Initial conditions
        st.subheader("🎲 Initial Conditions")
        c0 = st.slider("Average Li concentration", 0.1, 0.9, 0.5, 0.01)
        noise = st.slider("Fluctuation amplitude", 0.001, 0.1, 0.05, 0.001)
        
        if st.button("Apply ICs", use_container_width=True):
            sim.initialize_random(c0=c0, noise_amplitude=noise)
            st.rerun()
        
        st.divider()
        
        # Real-time statistics
        stats = sim.get_statistics()
        st.subheader("📊 Live Statistics")
        st.metric("Physical Time", stats['time_formatted'])
        st.metric("Simulation Step", f"{stats['step']:,}")
        st.metric("Domain Size", stats['domain_size'])
        st.metric("Interface Width", stats['interface_width'])
        st.metric("Diffusion Length", stats['diffusion_length'])
        st.metric("⟨c⟩", f"{stats['mean_concentration']:.3f}")
        st.metric("σ(c)", f"{stats['std_concentration']:.3f}")
        st.metric("Li-rich phase", f"{stats['phase_fraction_high']*100:.1f}%")
    
    # Main visualization area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Concentration Field (t = {stats['time_formatted']})")
        
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        im1 = ax1.imshow(sim.c, cmap='bwr', origin='lower', vmin=0, vmax=1,
                        extent=[0, sim.nx*sim.scales.L0*1e9, 0, sim.ny*sim.scales.L0*1e9])
        ax1.set_xlabel("x (nm)")
        ax1.set_ylabel("y (nm)")
        ax1.set_title(f"Li Concentration in LiₓFePO₄")
        cbar1 = plt.colorbar(im1, ax=ax1, label="Li fraction x")
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("Free Energy Density")
        energy = sim.compute_free_energy_density()
        
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(energy, cmap='viridis', origin='lower',
                        extent=[0, sim.nx*sim.scales.L0*1e9, 0, sim.ny*sim.scales.L0*1e9])
        ax2.set_xlabel("x (nm)")
        ax2.set_ylabel("y (nm)")
        cbar2 = plt.colorbar(im2, ax=ax2, label="J/m³")
        st.pyplot(fig2)
        plt.close(fig2)
        
        st.subheader("Concentration Histogram")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.hist(sim.c.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(stats['mean_concentration'], color='red', linestyle='--', 
                   label=f"⟨c⟩ = {stats['mean_concentration']:.2f}")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Li concentration x")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)
    
    # Time evolution plots
    st.subheader("📈 Kinetics")
    
    if len(sim.history['time_phys']) > 1:
        fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Convert time array to formatted labels for readability
        times_h = np.array(sim.history['time_phys']) / 3600  # hours
        
        # Mean concentration
        axes[0].plot(times_h, sim.history['mean'], 'b-', linewidth=2)
        axes[0].set_xlabel("Time (hours)")
        axes[0].set_ylabel("⟨c⟩")
        axes[0].set_title("Average Concentration")
        axes[0].grid(True, alpha=0.3)
        
        # Order parameter (std dev)
        axes[1].plot(times_h, sim.history['std'], 'r-', linewidth=2)
        axes[1].set_xlabel("Time (hours)")
        axes[1].set_ylabel("σ(c)")
        axes[1].set_title("Phase Separation Progress")
        axes[1].grid(True, alpha=0.3)
        
        # Phase fractions
        axes[2].plot(times_h, np.array(sim.history['phase_high'])*100, 
                    'g-', label='Li-rich (x>0.5)', linewidth=2)
        axes[2].plot(times_h, np.array(sim.history['phase_low'])*100, 
                    'orange', label='Li-poor (x<0.5)', linewidth=2)
        axes[2].set_xlabel("Time (hours)")
        axes[2].set_ylabel("Phase Fraction (%)")
        axes[2].set_title("Phase Evolution")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
    
    # Export section
    st.divider()
    st.subheader("💾 Export")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        if st.button("📸 Save Snapshot"):
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(sim.c, cmap='bwr', origin='lower', vmin=0, vmax=1,
                          extent=[0, sim.nx*sim.scales.L0*1e9, 0, sim.ny*sim.scales.L0*1e9])
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")
            ax.set_title(f"LiₓFePO₄ at t = {stats['time_formatted']}")
            plt.colorbar(im, ax=ax, label="Li fraction x")
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            st.download_button("Download PNG", buf.getvalue(), 
                             f"LiFePO4_t{sim.time_dim*sim.scales.t0:.1e}s.png", "image/png")
    
    with col_exp2:
        if st.button("📊 Save Statistics"):
            csv = "time_s,time_h,mean_c,std_c,phase_high,phase_low\n"
            for i in range(len(sim.history['time_phys'])):
                csv += f"{sim.history['time_phys'][i]},{sim.history['time_phys'][i]/3600},"
                csv += f"{sim.history['mean'][i]},{sim.history['std'][i]},"
                csv += f"{sim.history['phase_high'][i]},{sim.history['phase_low'][i]}\n"
            st.download_button("Download CSV", csv, "phase_field_stats.csv", "text/csv")
    
    with col_exp3:
        if st.button("⚙️ Save Parameters"):
            params = f"""# LiFePO4 Phase-Field Parameters
L0_nm = {sim.scales.L0*1e9:.2f}
D_b_m2s = {sim.scales.D_b:.2e}
Omega_kJmol = {sim.scales.Omega/1e3:.2f}
W_Jm3 = {sim.W_phys:.2e}
kappa_Jm = {sim.kappa_phys:.2e}
M_m5Js = {sim.M_phys:.2e}
dt_s = {sim.dt_phys:.2e}
interface_width_nm = {sim.scales.phys_to_interface_width(sim.kappa_phys, sim.W_phys)*1e9:.2f}
            """
            st.download_button("Download .txt", params, "simulation_params.txt", "text/plain")
    
    # Info section
    with st.expander("ℹ️ Physics & Usage Guide"):
        st.markdown("""
        ## 🔋 LiₓFePO₄ Phase-Field Model
        
        Simulates **spinodal decomposition** and **nucleation** in lithium iron phosphate cathodes.
        
        ### Governing Equation (Cahn-Hilliard)
        ```
        ∂c/∂t = ∇·[M ∇(∂f/∂c - κ∇²c)]
        
        f(c) = W·c²(1-c)²  [Regular solution free energy]
        ```
        
        ### Key Physical Parameters
        
        | Parameter | Symbol | Typical Range | Meaning |
        |-----------|--------|--------------|---------|
        | Reference length | L₀ | 5-100 nm | Interface/grid scale |
        | Diffusion coeff. | D_b | 10⁻¹⁶–10⁻¹² m²/s | Li⁺ mobility in crystal |
        | Mixing enthalpy | Ω | 5-60 kJ/mol | Phase separation driving force |
        | Gradient energy | κ | 10⁻¹¹–10⁻⁹ J/m | Interface energy penalty |
        | Mobility | M | 10⁻²⁰–10⁻¹⁶ m⁵/J·s | Kinetic coefficient |
        
        ### Interpreting Results
        
        - **Interface width**: ξ ≈ √(κ/W) — typical 2-10 nm for LiFePO₄
        - **Diffusion length**: ℓ_D = √(D·t) — how far Li diffuses in time t
        - **Phase separation time**: τ ≈ L²/D — coarsening timescale
        
        ### Stability Tips ⚠️
        
        1. Keep Δt small: Δt ≲ 0.1·L₀⁴/(M·κ) for explicit scheme stability
        2. Interface should span 3-5 grid points: adjust κ/W ratio
        3. Use "Reset" after major parameter changes
        
        ### Applications
        
        ✓ Study effect of temperature (via Ω, D_b) on phase separation  
        ✓ Optimize particle size (L₀) for fast charging  
        ✓ Compare spinodal vs nucleation mechanisms  
        ✓ Educational tool for phase-field methods  
        """)
    
    # Auto-run with physical time display
    st.sidebar.divider()
    auto_run = st.sidebar.checkbox("🔄 Auto-run", value=False)
    speed = st.sidebar.slider("Speed (steps/sec)", 1, 100, 10)
    
    if auto_run:
        placeholder = st.empty()
        if st.sidebar.button("⏹️ Stop"):
            auto_run = False
        
        with placeholder:
            for _ in range(speed):
                sim.run_step()
            st.rerun()


if __name__ == "__main__":
    main()
