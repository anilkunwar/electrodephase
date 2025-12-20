import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from io import BytesIO
import time

# =====================================================
# Numba-accelerated Phase Field Simulation Functions
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

class PhaseFieldSimulation:
    def __init__(self, nx=256, ny=256, dx=1.0, dt=0.1):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
        
        self.W = 1.0
        self.A = self.W
        self.B = -2.0 * self.W
        self.C = self.W
        
        self.kappa = 2.0
        self.M = 1.0
        
        self.c = np.zeros((nx, ny))
        self.time = 0.0
        self.step = 0
        self.history = {
            'time': [],
            'mean': [],
            'std': [],
            'phase_high': [],
            'phase_low': []
        }
        
    def set_parameters(self, W=None, kappa=None, M=None, A=None, B=None, C=None):
        if W is not None:
            self.W = W
            self.A = W
            self.B = -2.0 * W
            self.C = W
        
        if A is not None:
            self.A = A
        if B is not None:
            self.B = B
        if C is not None:
            self.C = C
        
        if kappa is not None:
            self.kappa = kappa
        if M is not None:
            self.M = M
    
    def initialize_random(self, c0=0.5, noise_amplitude=0.01):
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.time = 0.0
        self.step = 0
        self.clear_history()
    
    def initialize_seed(self, c0=0.3, seed_value=0.7, radius=15):
        self.c = c0 * np.ones((self.nx, self.ny))
        
        center_x, center_y = self.nx // 2, self.ny // 2
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                    self.c[i, j] = seed_value
        
        self.time = 0.0
        self.step = 0
        self.clear_history()
    
    def clear_history(self):
        self.history = {
            'time': [],
            'mean': [],
            'std': [],
            'phase_high': [],
            'phase_low': []
        }
        self.update_history()
    
    def update_history(self):
        self.history['time'].append(self.time)
        self.history['mean'].append(np.mean(self.c))
        self.history['std'].append(np.std(self.c))
        self.history['phase_high'].append(np.sum(self.c > 0.5) / (self.nx * self.ny))
        self.history['phase_low'].append(np.sum(self.c < 0.5) / (self.nx * self.ny))
    
    def run_step(self):
        self.c = update_concentration(
            self.c, self.dt, self.dx, 
            self.kappa, self.M,
            self.A, self.B, self.C
        )
        self.time += self.dt
        self.step += 1
        self.update_history()
    
    def run_steps(self, n_steps):
        for _ in range(n_steps):
            self.run_step()
    
    def compute_free_energy_density(self):
        energy = np.zeros_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                energy[i, j] = double_well_energy(self.c[i, j], self.A, self.B, self.C)
        return energy
    
    def get_statistics(self):
        return {
            'time': self.time,
            'step': self.step,
            'mean_concentration': np.mean(self.c),
            'std_concentration': np.std(self.c),
            'min_concentration': np.min(self.c),
            'max_concentration': np.max(self.c),
            'phase_fraction_high': np.sum(self.c > 0.5) / (self.nx * self.ny),
            'phase_fraction_low': np.sum(self.c < 0.5) / (self.nx * self.ny),
        }

# =====================================================
# Streamlit App
# =====================================================

def main():
    st.set_page_config(
        page_title="Phase Field Simulation",
        page_icon="🔬",
        layout="wide"
    )
    
    # Title and description
    st.title("🔬 Phase Field Simulation: Spinodal Decomposition")
    st.markdown("""
    This interactive simulation demonstrates **phase decomposition** using the Cahn-Hilliard equation.
    Adjust parameters to see how they affect phase separation dynamics.
    """)
    
    # Initialize simulation in session state
    if 'sim' not in st.session_state:
        st.session_state.sim = PhaseFieldSimulation(nx=256, ny=256, dx=1.0, dt=0.1)
        st.session_state.sim.initialize_random(c0=0.5, noise_amplitude=0.05)
    
    sim = st.session_state.sim
    
    # Sidebar for controls
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        
        st.subheader("Simulation Parameters")
        steps_to_run = st.number_input("Steps per update", min_value=1, max_value=1000, value=10)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running simulation..."):
                    sim.run_steps(steps_to_run)
        
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.rerun()
        
        if st.button("🔄 Reset Random", use_container_width=True):
            sim.initialize_random(c0=0.5, noise_amplitude=0.05)
            st.rerun()
        
        if st.button("🌱 Reset Seed", use_container_width=True):
            sim.initialize_seed(c0=0.3, seed_value=0.7, radius=15)
            st.rerun()
        
        st.divider()
        
        st.subheader("Free Energy Parameters")
        
        use_standard_double_well = st.checkbox("Use standard double-well (f(c)=W*c²(1-c)²)", value=True)
        
        if use_standard_double_well:
            W = st.slider("Double-well height (W)", 0.1, 5.0, 1.0, 0.1,
                         help="Controls barrier between phases. Higher = sharper interfaces.")
            sim.set_parameters(W=W, A=None, B=None, C=None)
        else:
            colA, colB, colC = st.columns(3)
            with colA:
                A = st.slider("A coefficient", 0.1, 5.0, 1.0, 0.1)
            with colB:
                B = st.slider("B coefficient", -5.0, 0.0, -2.0, 0.1)
            with colC:
                C = st.slider("C coefficient", 0.1, 5.0, 1.0, 0.1)
            sim.set_parameters(A=A, B=B, C=C)
        
        st.divider()
        
        st.subheader("Physical Parameters")
        kappa = st.slider("Gradient coefficient (κ)", 0.1, 10.0, 2.0, 0.1,
                         help="Controls interface width. Higher = wider interfaces.")
        M = st.slider("Mobility (M)", 0.01, 5.0, 1.0, 0.01,
                     help="Controls kinetics. Higher = faster phase separation.")
        dt = st.slider("Time step (Δt)", 0.01, 0.5, 0.1, 0.01,
                      help="Numerical time step. Too large may cause instability.")
        
        sim.kappa = kappa
        sim.M = M
        sim.dt = dt
        
        st.divider()
        
        st.subheader("Initial Conditions")
        c0 = st.slider("Average concentration", 0.1, 0.9, 0.5, 0.01,
                      help="Initial average concentration of the system.")
        noise = st.slider("Noise amplitude", 0.001, 0.1, 0.05, 0.001,
                         help="Initial random fluctuations.")
        
        if st.button("Apply Initial Conditions", use_container_width=True):
            sim.initialize_random(c0=c0, noise_amplitude=noise)
            st.rerun()
        
        st.divider()
        
        # Display statistics
        stats = sim.get_statistics()
        st.subheader("📊 Current Statistics")
        st.metric("Time", f"{stats['time']:.1f}")
        st.metric("Step", f"{stats['step']}")
        st.metric("Mean Concentration", f"{stats['mean_concentration']:.4f}")
        st.metric("Std Dev", f"{stats['std_concentration']:.4f}")
        st.metric("High Phase Fraction", f"{stats['phase_fraction_high']:.3f}")
        st.metric("Low Phase Fraction", f"{stats['phase_fraction_low']:.3f}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Concentration Field")
        
        # Create figure for concentration field
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        im1 = ax1.imshow(sim.c, cmap='bwr', origin='lower', vmin=0, vmax=1)
        ax1.set_title(f"Concentration Field (t = {sim.time:.1f})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1, label="Concentration c")
        st.pyplot(fig1)
        plt.close(fig1)
    
    with col2:
        st.subheader("Free Energy Density")
        
        # Create figure for free energy
        energy = sim.compute_free_energy_density()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(energy, cmap='viridis', origin='lower')
        ax2.set_title("Free Energy Density")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.colorbar(im2, ax=ax2, label="Energy Density")
        st.pyplot(fig2)
        plt.close(fig2)
        
        # Histogram
        st.subheader("Concentration Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.hist(sim.c.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Concentration c")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Histogram")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close(fig3)
    
    # Time series plots
    st.subheader("📈 Time Evolution")
    
    if len(sim.history['time']) > 1:
        fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Mean concentration
        axes[0].plot(sim.history['time'], sim.history['mean'], 'b-', linewidth=2)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Mean Concentration")
        axes[0].set_title("Mean Concentration vs Time")
        axes[0].grid(True, alpha=0.3)
        
        # Standard deviation
        axes[1].plot(sim.history['time'], sim.history['std'], 'r-', linewidth=2)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Standard Deviation")
        axes[1].set_title("Standard Deviation vs Time")
        axes[1].grid(True, alpha=0.3)
        
        # Phase fractions
        axes[2].plot(sim.history['time'], sim.history['phase_high'], 'g-', 
                    label='High phase (c > 0.5)', linewidth=2)
        axes[2].plot(sim.history['time'], sim.history['phase_low'], 'orange', 
                    label='Low phase (c < 0.5)', linewidth=2)
        axes[2].set_xlabel("Time")
        axes[2].set_ylabel("Phase Fraction")
        axes[2].set_title("Phase Fractions vs Time")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)
    
    # Export data section
    st.divider()
    st.subheader("💾 Export Data")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        if st.button("Save Current State as PNG"):
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(sim.c, cmap='bwr', origin='lower', vmin=0, vmax=1)
            ax.set_title(f"Phase Field Simulation - Time = {sim.time:.1f}")
            plt.colorbar(im, ax=ax, label="Concentration")
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            st.download_button(
                label="Download PNG",
                data=buf.getvalue(),
                file_name=f"phase_field_t{sim.time:.1f}.png",
                mime="image/png"
            )
    
    with col_exp2:
        if st.button("Save Statistics as CSV"):
            # Create CSV data
            csv_data = "time,mean_concentration,std_concentration,phase_fraction_high,phase_fraction_low\n"
            for i in range(len(sim.history['time'])):
                csv_data += f"{sim.history['time'][i]},{sim.history['mean'][i]},{sim.history['std'][i]},{sim.history['phase_high'][i]},{sim.history['phase_low'][i]}\n"
            
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="phase_field_statistics.csv",
                mime="text/csv"
            )
    
    # Information section
    with st.expander("ℹ️ About this Simulation"):
        st.markdown("""
        ## Phase Field Method for Spinodal Decomposition
        
        This simulation implements the **Cahn-Hilliard equation** to model phase separation in binary systems:
        
        ```
        ∂c/∂t = ∇·[M ∇μ]
        μ = df/dc - κ∇²c
        f(c) = A·c² + B·c³ + C·c⁴
        ```
        
        ### Key Parameters:
        
        **Double-well free energy (f(c))**:
        - **W** or **A, B, C**: Controls the energy barrier between phases
        - Higher barrier → sharper interfaces, stronger phase separation
        
        **Gradient coefficient (κ)**:
        - Controls interface width and energy
        - Higher κ → wider interfaces, higher interfacial energy
        
        **Mobility (M)**:
        - Controls kinetics of phase separation
        - Higher M → faster evolution
        
        ### Physical Interpretation:
        
        1. **Spinodal Decomposition**: When initialized with random fluctuations (average c=0.5), 
           the system undergoes spontaneous phase separation into interconnected patterns.
        
        2. **Nucleation and Growth**: When initialized with a seed (Reset Seed button), 
           a nucleus of the high-concentration phase grows in the low-concentration matrix.
        
        3. **Phase Coarsening**: Over time, smaller domains merge to form larger ones,
           reducing interfacial energy.
        
        ### Applications:
        - Binary alloys phase separation
        - Polymer blends
        - Battery electrode materials (LiFePO₄)
        - Pattern formation in materials science
        """)
    
    # Auto-run option
    st.sidebar.divider()
    auto_run = st.sidebar.checkbox("Auto-run simulation", value=False)
    auto_steps = st.sidebar.slider("Auto-run steps per second", 1, 100, 10)
    
    if auto_run:
        placeholder = st.empty()
        stop_button = st.sidebar.button("Stop Auto-run")
        
        if not stop_button:
            with placeholder.container():
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(auto_steps):
                    if stop_button:
                        break
                    
                    sim.run_step()
                    progress_bar.progress((i + 1) / auto_steps)
                    status_text.text(f"Running step {sim.step}...")
                    
                    # Update display every 5 steps
                    if (i + 1) % 5 == 0:
                        st.rerun()
                
                st.rerun()

if __name__ == "__main__":
    main()
