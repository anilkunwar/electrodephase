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
# PHYSICAL CONSTANTS AND SCALES FOR LiFePO₄
# =====================================================
class PhysicalScales:
    """Physical scales for LiFePO₄"""
   
    # Fundamental constants
    R = 8.314462618 # J/(mol·K)
    F = 96485.33212 # C/mol
    k_B = 1.380649e-23 # J/K
    e = 1.60217662e-19 # C (elementary charge)
    N_A = 6.02214076e23 # /mol
   
    def __init__(self):
        # Material properties
        self.T = 298.15 # K - Temperature
       
        # LiFePO₄ phase compositions
        self.c_alpha = 0.03 # FePO₄ phase
        self.c_beta = 0.97 # LiFePO₄ phase
       
        # Molar volume
        self.V_m = 3.0e-5 # m³/mol
       
        # Diffusion coefficient
        self.D_b = 1.0e-12 # m²/s - Adjusted for nano, from Bai refs ~10^-8 cm²/s = 10^-12 m²/s
       
        # Charge properties
        self.z = 1.0 # Li⁺ charge number
       
        # Regular solution parameter (adjusted to Bai: Ω = 0.183 eV/site)
        self.Omega_ev = 0.183  # eV per site
        self.kT_ev = self.k_B * self.T / self.e  # ~0.0257 eV
        self.Omega_tilde = self.Omega_ev / self.kT_ev  # ~7.12 dimensionless
        self.Omega = self.Omega_ev * self.e * self.N_A  # J/mol ~17.65 kJ/mol (original 55e3)
       
        # Set characteristic scales
        self.set_scales()
       
        # Kinetics parameters (from Bai/Hamadi)
        self.alpha = 0.5  # BV symmetry
        self.lambda_mhc = 8.3  # MHC reorganizational, dimensionless λ/RT
        self.tau_mhc = 3.358  # MHC correction
        self.k0 = 2.062e-4  # s^-1 rate constant from Hamadi
       
        print(f"Physical scales:")
        print(f" Length scale L0 = {self.L0:.2e} m ({self.L0*1e9:.1f} nm)")
        print(f" Time scale t0 = {self.t0:.2e} s")
        print(f" Dimensionless Omega_tilde = {self.Omega_tilde:.2f}")
       
    def set_scales(self):
        """Set characteristic scales"""
        # Length scale: 100 nm from Bai
        self.L0 = 1.0e-7 # 100 nm
       
        # Energy density scale from regular solution
        self.E0 = self.Omega / self.V_m # J/m³
       
        # Time scale from diffusion
        self.t0 = (self.L0**2) / self.D_b # s
       
        # Mobility scale
        self.M0 = self.D_b / (self.E0 * self.t0) # m⁵/(J·s)
       
        # Electric potential scale (thermal voltage)
        self.phi0 = self.R * self.T / self.F # ~0.0257 V at 298K
       
        # Exchange current scale (estimate from Bai ~10 mA/cm2 physical)
        self.J0_phys = 10e-3 / 1e-4  # 10 mA/cm2 = 0.1 A/m2, but adjust
        self.J0 = self.J0_phys * self.t0 / (self.F / self.N_A)  # dimensionless ~J0
       
    def dimensionless_to_physical(self, W_dim, kappa_dim, M_dim, dt_dim):
        """Convert dimensionless to physical"""
        W_phys = W_dim * self.E0
        kappa_phys = kappa_dim * self.E0 * self.L0**2
        M_phys = M_dim * self.M0
        dt_phys = dt_dim * self.t0
        return W_phys, kappa_phys, M_phys, dt_phys

# =====================================================
# NUMBA-ACCELERATED FUNCTIONS
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
def compute_laplacian(field, dx):
    """Compute 5-point stencil Laplacian with periodic in y, Neumann in x"""
    nx, ny = field.shape
    lap = np.zeros_like(field)
   
    for i in prange(nx):
        for j in prange(ny):
            jm1 = (j - 1) % ny
            jp1 = (j + 1) % ny
            
            if i == 0:  # Left boundary, one-sided
                lap[i, j] = (field[i+1, j] + field[i, j] + 
                             field[i, jp1] + field[i, jm1] - 4.0 * field[i, j]) / (dx * dx)
            elif i == nx-1:  # Right boundary, no-flux, one-sided
                lap[i, j] = (field[i, j] + field[i-1, j] +
                             field[i, jp1] + field[i, jm1] - 4.0 * field[i, j]) / (dx * dx)
            else:
                im1 = i - 1
                ip1 = i + 1
                lap[i, j] = (field[ip1, j] + field[im1, j] +
                             field[i, jp1] + field[i, jm1] - 4.0 * field[i, j]) / (dx * dx)
    return lap

@njit(fastmath=True, parallel=True)
def compute_gradient_x(field, dx):
    """Compute x-gradient with no-flux BCs in x, periodic y"""
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
   
    for i in prange(nx):
        for j in prange(ny):
            if i == 0:
                grad_x[i, j] = (field[i+1, j] - field[i, j]) / dx  # forward
            elif i == nx-1:
                grad_x[i, j] = (field[i, j] - field[i-1, j]) / dx  # backward
            else:
                grad_x[i, j] = (field[i+1, j] - field[i-1, j]) / (2.0 * dx)
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

@njit(fastmath=True)
def bv_flux(eta_tilde, c, alpha, J0_tilde):
    """Dimensionless BV flux"""
    return J0_tilde * (np.exp(-alpha * eta_tilde) - np.exp((1 - alpha) * eta_tilde))

@njit(fastmath=True)
def mhc_flux(eta_tilde, lambda_mhc, tau, J0_tilde):
    """Dimensionless MHC flux (approx from Hamadi/Bazant)"""
    exp_term = (np.exp(eta_tilde) - np.exp(-eta_tilde)) / (np.exp(eta_tilde) + np.exp(-eta_tilde) + 2)
    erfc_arg = (lambda_mhc - 1 + eta_tilde**2 / (2 * np.sqrt(lambda_mhc))) / (1 + np.sqrt(lambda_mhc))  # simplified
    erfc_val = 1 / (1 + np.exp(erfc_arg))  # simple sigmoid approx for erfc
    return tau * J0_tilde * np.sqrt(np.pi * lambda_mhc) * exp_term * erfc_val  # rough approx; refine if needed

@njit(fastmath=True)
def find_eta(I_tilde_target, c_boundary, kinetics_type, alpha, lambda_mhc, tau, J0_tilde, ny):
    """Bisection to find eta_tilde for average J = I_tilde / ny"""
    eta_low = -10.0
    eta_high = 10.0
    tol = 1e-6
    max_iter = 100
    
    for _ in range(max_iter):
        eta_mid = (eta_low + eta_high) / 2.0
        J_sum = 0.0
        for j in range(ny):
            if kinetics_type == 0:  # BV
                J = bv_flux(eta_mid, c_boundary[j], alpha, J0_tilde)
            else:  # MHC
                J = mhc_flux(eta_mid, lambda_mhc, tau, J0_tilde)
            J_sum += J
        J_avg = J_sum / ny
        if abs(J_avg - I_tilde_target / ny) < tol:
            return eta_mid
        if J_avg < I_tilde_target / ny:
            eta_low = eta_mid
        else:
            eta_high = eta_mid
    return eta_mid  # approximate if not converged

@njit(fastmath=True, parallel=True)
def update_concentration(c, dt, dx, kappa, M, A, B, C, eta_tilde, kinetics_type, alpha, lambda_mhc, tau, J0_tilde, I_tilde):
    """Update concentration with diffusive flux only; add kinetics at left boundary i=0"""
    nx, ny = c.shape
   
    # Compute Laplacian of concentration
    lap_c = compute_laplacian(c, dx)
   
    # Chemical potential from free energy
    mu_chem = chemical_potential(c, A, B, C) - kappa * lap_c
   
    # Compute gradients
    mu_grad_x = compute_gradient_x(mu_chem, dx)
    mu_grad_y = compute_gradient_y(mu_chem, dx)
   
    # Diffusive flux: -M ∇μ
    flux_x = -M * mu_grad_x
    flux_y = -M * mu_grad_y
   
    # Compute divergence of flux
    div_flux = np.zeros_like(c)
   
    for i in prange(nx):
        for j in prange(ny):
            if i == nx-1:
                div_x = (flux_x[nx-1, j] - flux_x[nx-2, j]) / dx  # backward at right
            elif i == 0:
                div_x = (flux_x[1, j] - flux_x[0, j]) / dx  # forward at left
            else:
                div_x = (flux_x[i+1, j] - flux_x[i-1, j]) / (2.0 * dx)
            
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            div_y = (flux_y[i, jp1] - flux_y[i, jm1]) / (2.0 * dx)
           
            div_flux[i, j] = div_x + div_y
   
    # Update concentration in bulk: ∂c/∂t = -∇·flux
    c_new = c - dt * div_flux
   
    # Add kinetics flux at left boundary (i=0)
    c_boundary = c_new[0, :]
    eta_tilde = find_eta(I_tilde, c_boundary, kinetics_type, alpha, lambda_mhc, tau, J0_tilde, ny)
    for j in prange(ny):
        if kinetics_type == 0:  # BV
            J = bv_flux(eta_tilde, c_new[0, j], alpha, J0_tilde)
        else:  # MHC
            J = mhc_flux(eta_tilde, lambda_mhc, tau, J0_tilde)
        c_new[0, j] += dt * J / dx  # add flux, sign for insertion (lithiation); flip for delith if needed
   
    # Ensure concentration stays in [0, 1]
    c_new = np.minimum(1.0, np.maximum(0.0, c_new))
   
    return c_new, eta_tilde

# =====================================================
# PHASE FIELD SIMULATION
# =====================================================
class PhaseFieldSimulation:
    """Phase field simulation for LiFePO₄ with kinetics"""
   
    def __init__(self, nx=256, ny=256, dx=1.0, dt=0.1):
        # Simulation grid
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt
       
        # Physical scales
        self.scales = PhysicalScales()
       
        # Dimensionless parameters (adjusted to Bai)
        self.W_dim = self.scales.Omega_tilde / 4  # Approx for double-well height ~ Omega/4
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
       
        self.kappa_dim = 0.00266  # From Bai ~K approx
        self.M_dim = 1.0
       
        # Update physical
        self.update_physical_parameters()
       
        # Fields
        self.c = np.zeros((nx, ny)) # Concentration
       
        # Kinetics
        self.kinetics_type = 0  # 0 BV, 1 MHC
        self.I_tilde = 0.01  # Dimensionless current from Bai
        self.eta_tilde = 0.0
       
        # Time tracking
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
       
        # History
        self.history = {
            'time_phys': [],
            'mean_c': [],
            'std_c': [],
            'voltage': [],
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'eta_tilde': []
        }
       
        # Initialize
        self.initialize_random()
   
    def update_physical_parameters(self):
        self.W_phys, self.kappa_phys, self.M_phys, self.dt_phys = \
            self.scales.dimensionless_to_physical(
                self.W_dim, self.kappa_dim, self.M_dim, self.dt
            )
        self.A = self.W_dim
        self.B = -2.0 * self.W_dim
        self.C = self.W_dim
        self.J0_tilde = 1.0  # Assume unit for dimless, adjust if needed
   
    def set_parameters(self, W_dim=None, kappa_dim=None, M_dim=None, dt_dim=None, I_tilde=None, kinetics_type=None):
        if W_dim is not None:
            self.W_dim = W_dim
        if kappa_dim is not None:
            self.kappa_dim = kappa_dim
        if M_dim is not None:
            self.M_dim = M_dim
        if dt_dim is not None:
            self.dt = dt_dim
        if I_tilde is not None:
            self.I_tilde = I_tilde
        if kinetics_type is not None:
            self.kinetics_type = 0 if kinetics_type == "BV" else 1
        self.update_physical_parameters()
   
    def initialize_random(self, c0=0.5, noise_amplitude=0.05):
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
   
    def initialize_lithiation(self):
        self.c = self.scales.c_alpha * np.ones((self.nx, self.ny))
        seed_width = self.ny // 10
        for j in range(self.ny):
            if abs(j - self.ny//2) < seed_width:
                self.c[:10, j] = self.scales.c_beta
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
   
    def initialize_delithiation(self):
        self.c = self.scales.c_beta * np.ones((self.nx, self.ny))
        seed_width = self.ny // 10
        for j in range(self.ny):
            if abs(j - self.ny//2) < seed_width:
                self.c[:10, j] = self.scales.c_alpha
        self.time_dim = 0.0
        self.time_phys = 0.0
        self.step = 0
        self.clear_history()
   
    def clear_history(self):
        self.history = {
            'time_phys': [],
            'mean_c': [],
            'std_c': [],
            'voltage': [],
            'phase_FePO4': [],
            'phase_LiFePO4': [],
            'eta_tilde': []
        }
        self.update_history()
   
    def update_history(self):
        self.history['time_phys'].append(self.time_phys)
        self.history['mean_c'].append(np.mean(self.c))
        self.history['std_c'].append(np.std(self.c))
        voltage = self.scales.phi0 * (self.eta_tilde + np.mean(self.compute_chemical_potential()) / self.scales.F)  # Approx V
        self.history['voltage'].append(voltage)
        threshold = 0.5
        self.history['phase_FePO4'].append(np.sum(self.c < threshold) / (self.nx * self.ny))
        self.history['phase_LiFePO4'].append(np.sum(self.c >= threshold) / (self.nx * self.ny))
        self.history['eta_tilde'].append(self.eta_tilde)
   
    def run_step(self):
        self.c, self.eta_tilde = update_concentration(
            self.c, self.dt, self.dx, self.kappa_dim, self.M_dim, self.A, self.B, self.C, self.eta_tilde,
            self.kinetics_type, self.scales.alpha, self.scales.lambda_mhc, self.scales.tau_mhc, self.J0_tilde, self.I_tilde
        )
        self.time_dim += self.dt
        self.time_phys += self.dt_phys
        self.step += 1
        self.update_history()
   
    def run_steps(self, n_steps):
        for _ in range(n_steps):
            self.run_step()
   
    def compute_chemical_potential(self):
        lap_c = compute_laplacian(self.c, self.dx)
        return chemical_potential(self.c, self.A, self.B, self.C) - self.kappa_dim * lap_c
   
    def compute_free_energy_density(self):
        energy = double_well_energy(self.c, self.A, self.B, self.C)
        grad_x = compute_gradient_x(self.c, self.dx)
        grad_y = compute_gradient_y(self.c, self.dx)
        grad_sq = grad_x**2 + grad_y**2
        energy += 0.5 * self.kappa_dim * grad_sq
        return energy
   
    def get_statistics(self):
        stats = {
            'time_phys': self.time_phys,
            'step': self.step,
            'mean_c': np.mean(self.c),
            'std_c': np.std(self.c),
            'x_Li': np.mean(self.c),
            'voltage': self.history['voltage'][-1] if self.history['voltage'] else 0.0,
            'phase_FePO4': np.sum(self.c < 0.5) / (self.nx * self.ny),
            'phase_LiFePO4': np.sum(self.c >= 0.5) / (self.nx * self.ny),
            'domain_size_nm': self.nx * self.dx * self.scales.L0 * 1e9,
            'interface_width_nm': np.sqrt(self.kappa_phys / self.W_phys) * 1e9,
            'W_dim': self.W_dim,
            'kappa_dim': self.kappa_dim,
            'M_dim': self.M_dim,
            'W_phys': self.W_phys,
            'kappa_phys': self.kappa_phys,
            'M_phys': self.M_phys,
            'dt_phys': self.dt_phys,
        }
        return stats

# =====================================================
# STREAMLIT APP
# =====================================================
def main():
    st.set_page_config(page_title="LiFePO₄ Phase Field with Kinetics", page_icon="⚡", layout="wide")
   
    st.title("⚡ LiFePO₄ Phase Field with BV/MHC Kinetics")
    st.markdown("""
    ### Cahn-Hilliard Model with Surface Kinetics for Battery Electrodes
    Enhanced with BV or MHC kinetics under constant current, inspired by Bai et al.
    """)
   
    if 'sim' not in st.session_state:
        st.session_state.sim = PhaseFieldSimulation(nx=256, ny=256, dx=1.0, dt=0.01)
   
    sim = st.session_state.sim
   
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
       
        with st.expander("Parameters", expanded=True):
            stats = sim.get_statistics()
            st.markdown(f"""
            **Domain Size:** {stats['domain_size_nm']:.0f} nm
            **Interface Width:** {stats['interface_width_nm']:.2f} nm
            **Omega_tilde:** {sim.scales.Omega_tilde:.2f}
            """)
       
        st.divider()
       
        col1, col2 = st.columns(2)
        with col1:
            steps = st.number_input("Steps/update", 1, 500, 10)
       
        with col2:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running..."):
                    sim.run_steps(steps)
                    st.rerun()
       
        st.divider()
       
        st.subheader("Electrochemical Scenarios")
        init_option = st.radio("Choose scenario:", ["Random (No Bias)", "Lithiation (Charge)", "Delithiation (Discharge)"], index=0)
       
        if st.button("🔄 Apply Scenario", use_container_width=True):
            if init_option == "Random (No Bias)":
                sim.initialize_random(c0=0.5, noise_amplitude=0.05)
            elif init_option == "Lithiation (Charge)":
                sim.initialize_lithiation()
            else:
                sim.initialize_delithiation()
            st.rerun()
       
        st.divider()
       
        st.subheader("Model Parameters")
        W_dim = st.slider("W (Double-well)", 0.1, 10.0, float(sim.W_dim), 0.1)
        kappa_dim = st.slider("κ (Gradient)", 0.001, 0.01, float(sim.kappa_dim), 0.001)
        M_dim = st.slider("M (Mobility)", 0.01, 5.0, float(sim.M_dim), 0.01)
        dt_dim = st.slider("Δt (Time step)", 0.001, 0.1, float(sim.dt), 0.001)
        I_tilde = st.select_slider("Dimensionless Current ~I (Bai-like)", options=[0.01, 0.25, 2.0], value=0.01)
        kinetics_option = st.selectbox("Kinetics Type", ["BV", "MHC"])
       
        sim.set_parameters(W_dim=W_dim, kappa_dim=kappa_dim, M_dim=M_dim, dt_dim=dt_dim, I_tilde=I_tilde, kinetics_type=kinetics_option)
       
        st.divider()
       
        st.subheader("📊 Current State")
        stats = sim.get_statistics()
       
        col_stat1, col2 = st.columns(2)
        with col_stat1:
            st.metric("Time", f"{stats['time_phys']:.2e} s")
            st.metric("x in LiₓFePO₄", f"{stats['mean_c']:.3f}")
            st.metric("Voltage", f"{stats['voltage']:.3f} V")
       
        with col2:
            st.metric("FePO₄", f"{stats['phase_FePO4']:.3f}")
            st.metric("LiFePO₄", f"{stats['phase_LiFePO4']:.3f}")
            st.metric("Overpotential ~η", f"{sim.history['eta_tilde'][-1]:.3f}" if sim.history['eta_tilde'] else 0.0)
   
    # Tabs for visualization (similar to original, removed electric field tab)
    tab1, tab2, tab3 = st.tabs(["Concentration", "Statistics", "Free Energy"])
   
    with tab1:
        # Concentration plot (similar)
        domain_nm = sim.nx * sim.dx * sim.scales.L0 * 1e9
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        im1 = ax1.imshow(sim.c.T, cmap='RdYlBu', origin='lower', vmin=0, vmax=1, aspect='auto')
        ax1.set_title(f"LiₓFePO₄ Concentration\nTime = {stats['time_phys']:.2e} s, x = {stats['mean_c']:.3f}")
        ax1.set_xlabel(f"x ({domain_nm:.0f} nm)")
        ax1.set_ylabel(f"y ({domain_nm:.0f} nm)")
        plt.colorbar(im1, ax=ax1, label='x in LiₓFePO₄')
        st.pyplot(fig1)
   
    with tab2:
        # Statistics (similar, added eta)
        if len(sim.history['time_phys']) > 1:
            fig4, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes[0, 0].plot(sim.history['time_phys'], sim.history['mean_c'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel("Time (s)")
            axes[0, 0].set_ylabel("Mean x")
            axes[0, 0].set_title("Lithium Content")
            axes[0, 0].grid(True)
           
            axes[0, 1].plot(sim.history['time_phys'], sim.history['voltage'], color='orange', linewidth=2)
            axes[0, 1].set_xlabel("Time (s)")
            axes[0, 1].set_ylabel("Voltage (V)")
            axes[0, 1].set_title("Voltage Evolution")
            axes[0, 1].grid(True)
           
            axes[1, 0].plot(sim.history['time_phys'], sim.history['phase_FePO4'], 'r-', label='FePO₄-rich', linewidth=2)
            axes[1, 0].plot(sim.history['time_phys'], sim.history['phase_LiFePO4'], 'g-', label='LiFePO₄-rich', linewidth=2)
            axes[1, 0].set_xlabel("Time (s)")
            axes[1, 0].set_ylabel("Phase Fraction")
            axes[1, 0].set_title("Phase Evolution")
            axes[1, 0].legend()
            axes[1, 0].grid(True)
           
            axes[1, 1].plot(sim.history['time_phys'], sim.history['eta_tilde'], color='purple', linewidth=2)
            axes[1, 1].set_xlabel("Time (s)")
            axes[1, 1].set_ylabel("~η")
            axes[1, 1].set_title("Overpotential Evolution")
            axes[1, 1].grid(True)
           
            plt.tight_layout()
            st.pyplot(fig4)
           
            df = pd.DataFrame({
                'Time (s)': sim.history['time_phys'],
                'Mean x': sim.history['mean_c'],
                'Std x': sim.history['std_c'],
                'Voltage (V)': sim.history['voltage'],
                'FePO₄ fraction': sim.history['phase_FePO4'],
                'LiFePO₄ fraction': sim.history['phase_LiFePO4'],
                '~η': sim.history['eta_tilde']
            })
            st.dataframe(df.tail(10))
            csv = df.to_csv(index=False)
            st.download_button(label="📥 Download CSV", data=csv, file_name="phase_field_kinetics.csv", mime="text/csv")
        else:
            st.info("Run simulation to see statistics.")
   
    with tab3:
        # Free energy density plot (enhanced)
        energy = sim.compute_free_energy_density()
        fig_en, ax_en = plt.subplots(figsize=(8, 7))
        im_en = ax_en.imshow(energy.T, cmap='viridis', origin='lower', aspect='auto')
        ax_en.set_title("Free Energy Density")
        ax_en.set_xlabel("x position")
        ax_en.set_ylabel("y position")
        plt.colorbar(im_en, ax=ax_en, label='Energy Density')
        st.pyplot(fig_en)
   
    with st.expander("📚 Governing Equations with Kinetics", expanded=True):
        st.markdown("""
        ### Cahn-Hilliard Equation
        ∂c/∂t = ∇·[M ∇μ]
        where μ = df/dc - κ∇²c
        
        ### Kinetics at Boundary
        For BV: J = J0 [exp(-α η) - exp((1-α) η)]
        For MHC: Approximate formula with λ=8.3, τ=3.358
        
        Constant current ~I enforced by adjusting η each step.
        """)

if __name__ == "__main__":
    main()
