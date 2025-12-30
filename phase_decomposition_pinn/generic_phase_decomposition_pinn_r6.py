import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st
from io import BytesIO
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import logging
from typing import Dict, Any

# =====================================================
# Configuration & Setup
# =====================================================

# Ensure output directory exists
OUTPUT_DIR = '/tmp/pinn_phase_field'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'pinn_training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Force CPU execution and limit threads for Streamlit Cloud stability
torch.set_num_threads(2)

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
# Memory-Safe PINN Model Definition
# =====================================================

class CahnHilliardPINN(nn.Module):
    """Physics-Informed Neural Network for Cahn-Hilliard equation with periodic BCs."""
    
    def __init__(self, Lx: float, Ly: float, T_max: float, 
                 A: float, B: float, C: float, 
                 kappa: float, M: float):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        self.A = A
        self.B = B
        self.C = C
        self.kappa = kappa
        self.M = M
        
        # Network architecture: Periodic embedding + MLP
        self.net = nn.Sequential(
            nn.Linear(5, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Enforce c ∈ [0, 1]
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization for better convergence."""
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass with periodic embedding."""
        # Normalize coordinates
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        
        # Periodic embedding: sin(2πx/Lx), cos(2πx/Lx), etc.
        x_sin = torch.sin(2 * np.pi * x_norm)
        x_cos = torch.cos(2 * np.pi * x_norm)
        y_sin = torch.sin(2 * np.pi * y_norm)
        y_cos = torch.cos(2 * np.pi * y_norm)
        
        # Concatenate all inputs
        inputs = torch.cat([x_sin, x_cos, y_sin, y_cos, t_norm], dim=1)
        c = self.net(inputs)
        return c
    
    def chemical_potential(self, c: torch.Tensor) -> torch.Tensor:
        """Compute chemical potential μ = df/dc."""
        return 2.0 * self.A * c + 3.0 * self.B * c**2 + 4.0 * self.C * c**3

# =====================================================
# PINN Physics Utilities
# =====================================================

def laplacian(u: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute ∇²u = u_xx + u_yy using automatic differentiation."""
    # First derivatives
    u_x = torch.autograd.grad(
        u, x, 
        grad_outputs=torch.ones_like(u),
        create_graph=True, 
        retain_graph=True
    )[0]
    u_y = torch.autograd.grad(
        u, y, 
        grad_outputs=torch.ones_like(u),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x, x, 
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, 
        retain_graph=True
    )[0]
    u_yy = torch.autograd.grad(
        u_y, y, 
        grad_outputs=torch.ones_like(u_y),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    return u_xx + u_yy

def pde_residual(model: CahnHilliardPINN, 
                x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute Cahn-Hilliard PDE residual."""
    # Predict concentration
    c = model(x, y, t)
    
    # Time derivative: ∂c/∂t
    c_t = torch.autograd.grad(
        c, t, 
        grad_outputs=torch.ones_like(c),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    # Laplacian of concentration: ∇²c
    lap_c = laplacian(c, x, y)
    
    # Chemical potential: μ = f'(c) - κ∇²c
    mu = model.chemical_potential(c) - model.kappa * lap_c
    
    # Laplacian of chemical potential: ∇²μ
    lap_mu = laplacian(mu, x, y)
    
    # Cahn-Hilliard residual: ∂c/∂t - M∇²μ
    residual = c_t - model.M * lap_mu
    return residual

def initial_condition_loss(model: CahnHilliardPINN, 
                          x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute initial condition loss with smooth noise."""
    # Create smooth random initial condition using low-frequency modes
    noise_x = torch.sin(2 * np.pi * x / model.Lx) * torch.cos(4 * np.pi * y / model.Ly)
    noise_y = torch.cos(4 * np.pi * x / model.Lx) * torch.sin(2 * np.pi * y / model.Ly)
    smooth_noise = 0.5 * (noise_x + noise_y)
    
    c_target = 0.5 + 0.05 * smooth_noise
    c_pred = model(x, y, torch.zeros_like(x))
    
    return torch.mean((c_pred - c_target)**2)

def generate_collocation_points(pde_points: int, ic_points: int,
                              Lx: float, Ly: float, T_max: float) -> Dict[str, torch.Tensor]:
    """Generate collocation points for PDE and initial condition."""
    # PDE points (interior)
    x_pde = torch.rand(pde_points, 1, requires_grad=True) * Lx
    y_pde = torch.rand(pde_points, 1, requires_grad=True) * Ly
    t_pde = torch.rand(pde_points, 1, requires_grad=True) * T_max
    
    # Initial condition points (t=0)
    x_ic = torch.rand(ic_points, 1, requires_grad=True) * Lx
    y_ic = torch.rand(ic_points, 1, requires_grad=True) * Ly
    
    return {
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_ic': x_ic, 'y_ic': y_ic
    }

# =====================================================
# MEMORY-SAFE TRAINING FUNCTION (KEY COMPONENT)
# =====================================================

def train_pinn_safe(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train PINN in complete isolation. Returns ONLY safe, graph-free data.
    No tensors with grad_fn are returned.
    """
    device = torch.device('cpu')  # Critical for Streamlit Cloud
    
    # Apply safety caps to prevent OOM
    pde_points = min(params.get('pde_points', 4000), 5000)
    ic_points = min(params.get('ic_points', 1500), 2000)
    epochs = min(params.get('epochs', 2000), 3000)
    T_max = min(params.get('T_max', 30.0), 40.0)
    
    Lx = params['Lx']
    Ly = params['Ly']
    W = params['W']
    kappa = params['kappa']
    M = params['M']
    lr = params.get('lr', 1e-3)
    
    # Set up free energy coefficients (standard double-well)
    A = W
    B = -2.0 * W
    C = W
    
    # Initialize model and move to CPU
    model = CahnHilliardPINN(
        Lx=Lx, Ly=Ly, T_max=T_max,
        A=A, B=B, C=C,
        kappa=kappa, M=M
    ).to(device)
    
    # Generate collocation points
    points = generate_collocation_points(pde_points, ic_points, Lx, Ly, T_max)
    points = {k: v.to(device) for k, v in points.items()}
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Loss history (store ONLY scalars)
    loss_history = {
        'epochs': [],
        'total': [],
        'pde': [],
        'ic': []
    }
    
    start_time = time.time()
    
    try:
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute PDE residual and losses
            residual = pde_residual(model, points['x_pde'], points['y_pde'], points['t_pde'])
            pde_loss = torch.mean(residual ** 2)
            ic_loss = initial_condition_loss(model, points['x_ic'], points['y_ic'])
            total_loss = pde_loss + 10.0 * ic_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store ONLY scalar values (no tensors!)
            if (epoch + 1) % 100 == 0:
                loss_history['epochs'].append(epoch + 1)
                loss_history['total'].append(total_loss.item())      # ← .item() is critical
                loss_history['pde'].append(pde_loss.item())
                loss_history['ic'].append(ic_loss.item())
            
            # Explicitly delete tensors to free autograd graph
            del residual, pde_loss, ic_loss, total_loss
        
        training_time = time.time() - start_time
        
        # Extract model state as pure NumPy arrays (safe to store)
        model_state = {}
        for k, v in model.state_dict().items():
            model_state[k] = v.detach().cpu().numpy()
        
        # Explicit cleanup
        del model, points, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "success": True,
            "model_state": model_state,
            "loss_history": loss_history,
            "training_time": training_time,
            "params": params
        }
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {"error": str(e)}

# =====================================================
# MEMORY-SAFE EVALUATION FUNCTIONS
# =====================================================

def load_pinn_model(model_state: Dict[str, np.ndarray], params: Dict[str, Any]) -> CahnHilliardPINN:
    """Reconstruct model from safe NumPy state dict."""
    A = params['W']
    B = -2.0 * params['W']
    C = params['W']
    
    model = CahnHilliardPINN(
        params['Lx'], params['Ly'], params['T_max'],
        A, B, C, params['kappa'], params['M']
    )
    
    # Load state dict from NumPy arrays
    state_dict = {}
    for k, v in model_state.items():
        state_dict[k] = torch.from_numpy(v)
    model.load_state_dict(state_dict)
    
    return model

def evaluate_pinn_safe(model_state: Dict[str, np.ndarray], params: Dict[str, Any], times: np.ndarray) -> Dict[str, Any]:
    """Evaluate the trained PINN on a grid without creating computation graphs."""
    model = load_pinn_model(model_state, params)
    model.eval()
    
    # Use lower resolution for memory safety
    nx = min(params.get('nx', 96), 128)
    ny = min(params.get('ny', 96), 128)
    Lx = params['Lx']
    Ly = params['Ly']
    
    # Create evaluation grid
    x = torch.linspace(0, Lx, nx)
    y = torch.linspace(0, Ly, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c_solutions = []
    free_energy_solutions = []
    
    # Evaluate at each time step
    with torch.no_grad():  # ← Critical: no autograd graph
        for t_val in times:
            t = torch.full((X.numel(), 1), t_val, dtype=torch.float32)
            c_pred = model(X.reshape(-1, 1), Y.reshape(-1, 1), t)
            c = c_pred.reshape(nx, ny).cpu().numpy()
            c_solutions.append(c)
            
            # Compute free energy density
            A = params['W']
            B = -2.0 * params['W']
            C = params['W']
            energy = A * c**2 + B * c**3 + C * c**4
            free_energy_solutions.append(energy)
    
    return {
        'c_solutions': c_solutions,
        'free_energy_solutions': free_energy_solutions,
        'times': times,
        'params': params
    }

# =====================================================
# STREAMLIT USER INTERFACE
# =====================================================

def add_pinn_section_safe():
    """Add the memory-safe PINN section to the Streamlit app."""
    st.header("🧠 Physics-Informed Neural Network (PINN) - Memory Safe")
    st.markdown("""
    This PINN solves the Cahn-Hilliard equation **without numerical discretization**.
    All computation graphs are destroyed after training — safe for Streamlit Cloud!
    """)
    
    # Demo mode toggle (recommended for first-time users)
    demo_mode = st.sidebar.checkbox("💡 Light Demo Mode (recommended)", True)
    
    with st.sidebar:
        st.divider()
        st.subheader("🧠 PINN Parameters")
        if demo_mode:
            pinn_epochs = 1500
            pde_points = 3500
            st.info("Light mode: epochs=1500, PDE points=3500")
        else:
            pinn_epochs = st.number_input("Training epochs", 500, 5000, 2000, 100)
            pde_points = st.number_input("PDE collocation points", 1000, 6000, 4500, 500)
    
    # Get current FDM parameters or use defaults
    if 'sim' in st.session_state:
        sim = st.session_state.sim
        base_params = {
            'Lx': float(sim.dx * sim.nx),
            'Ly': float(sim.dx * sim.ny),
            'W': sim.W,
            'kappa': sim.kappa,
            'M': sim.M,
        }
    else:
        base_params = {'Lx': 60.0, 'Ly': 60.0, 'W': 1.0, 'kappa': 2.0, 'M': 1.0}
    
    # Build full parameter set with safety limits
    current_params = {
        **base_params,
        'T_max': 25.0 if demo_mode else 35.0,
        'epochs': pinn_epochs,
        'lr': 1e-3,
        'pde_points': pde_points,
        'ic_points': 1500,
        'nx': 80 if demo_mode else 100,
        'ny': 80 if demo_mode else 100
    }
    
    st.info(f"Domain: {current_params['Lx']} × {current_params['Ly']} μm² | Simulation time: {current_params['T_max']} seconds")
    
    # Training button
    if st.button("🚀 Train PINN (CPU Only)", type="primary"):
        with st.spinner("Training PINN... This may take 1-5 minutes."):
            result = train_pinn_safe(current_params)
            
            if "error" in result:
                st.error(f"Training failed: {result['error']}")
            else:
                st.session_state.pinn_result = result
                st.success(f"✅ PINN training completed in {result['training_time']:.1f} seconds!")
    
    # Display results if available
    if 'pinn_result' in st.session_state:
        result = st.session_state.pinn_result
        
        # Training metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Time", f"{result['training_time']:.1f}s")
        col2.metric("Final PDE Loss", f"{result['loss_history']['pde'][-1]:.2e}")
        col3.metric("Final IC Loss", f"{result['loss_history']['ic'][-1]:.2e}")
        
        # Loss history plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result['loss_history']['epochs'], result['loss_history']['total'], 'k-', linewidth=2, label='Total Loss')
        ax.plot(result['loss_history']['epochs'], result['loss_history']['pde'], 'b--', linewidth=1.5, label='PDE Loss')
        ax.plot(result['loss_history']['epochs'], result['loss_history']['ic'], 'r-.', linewidth=1.5, label='IC Loss')
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('PINN Training Losses')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        # Evaluation button
        if st.button("🔍 Evaluate Solution at Multiple Times"):
            with st.spinner("Evaluating PINN solution..."):
                times = np.linspace(0, result['params']['T_max'], 8)
                solution = evaluate_pinn_safe(result['model_state'], result['params'], times)
                st.session_state.pinn_solution = solution
                st.success("Evaluation complete! Scroll down to view results.")
        
        # Display solution if available
        if 'pinn_solution' in st.session_state:
            sol = st.session_state.pinn_solution
            time_idx = st.slider(
                "Select time step", 
                0, len(sol['times']) - 1, 
                len(sol['times']) - 1,
                format="t = %.1f s"
            )
            t_val = sol['times'][time_idx]
            
            c = sol['c_solutions'][time_idx]
            energy = sol['free_energy_solutions'][time_idx]
            
            # Concentration and energy plots
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax1 = plt.subplots(figsize=(6, 5))
                im1 = ax1.imshow(c.T, cmap='bwr', origin='lower', vmin=0, vmax=1)
                ax1.set_title(f"PINN Concentration (t = {t_val:.1f}s)")
                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                plt.colorbar(im1, ax=ax1, label="Concentration c")
                st.pyplot(fig1)
                plt.close()
            with col2:
                fig2, ax2 = plt.subplots(figsize=(6, 5))
                im2 = ax2.imshow(energy.T, cmap='viridis', origin='lower')
                ax2.set_title(f"PINN Free Energy (t = {t_val:.1f}s)")
                ax2.set_xlabel("x")
                ax2.set_ylabel("y")
                plt.colorbar(im2, ax=ax2, label="Energy Density")
                st.pyplot(fig2)
                plt.close()
            
            # Concentration histogram
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            ax3.hist(c.flatten(), bins=40, alpha=0.7, color='steelblue', edgecolor='black')
            ax3.set_xlim(0, 1)
            ax3.set_xlabel("Concentration c")
            ax3.set_ylabel("Frequency")
            ax3.set_title(f"Concentration Distribution (t = {t_val:.1f}s)")
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close()
            
            # Download section
            st.subheader("💾 Download PINN Results")
            
            # Solution data
            solution_bytes = pickle.dumps(sol)
            st.download_button(
                "📥 Download Solution Data (Pickle)",
                solution_bytes,
                "pinn_solution.pkl",
                "application/octet-stream"
            )
            
            # Loss history
            loss_bytes = pickle.dumps(result['loss_history'])
            st.download_button(
                "📥 Download Loss History (Pickle)",
                loss_bytes,
                "pinn_loss_history.pkl",
                "application/octet-stream"
            )

# =====================================================
# MAIN APPLICATION
# =====================================================

def main():
    st.set_page_config(
        page_title="Phase Field + PINN Simulation",
        page_icon="🔬",
        layout="wide"
    )
    
    # Title and description
    st.title("🔬 Phase Field Simulation + Physics-Informed Neural Network")
    st.markdown("""
    **Top section**: Traditional finite-difference solver (fast, robust).  
    **Bottom section**: Mesh-free PINN that learns directly from physics (no data required).
    """)
    
    # Initialize FDM simulator in session state
    if 'sim' not in st.session_state:
        st.session_state.sim = PhaseFieldSimulation(nx=256, ny=256, dx=1.0, dt=0.1)
        st.session_state.sim.initialize_random(c0=0.5, noise_amplitude=0.05)
    
    sim = st.session_state.sim
    
    # Sidebar controls for FDM
    with st.sidebar:
        st.header("🎛️ FDM Controls")
        
        steps_to_run = st.number_input("Steps per update", min_value=1, max_value=1000, value=10)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run FDM Steps", use_container_width=True):
                with st.spinner("Running finite-difference simulation..."):
                    sim.run_steps(steps_to_run)
        
        with col2:
            if st.button("🔄 Reset FDM", use_container_width=True):
                sim.initialize_random(c0=0.5, noise_amplitude=0.05)
                st.rerun()
        
        st.divider()
        st.subheader("FDM Parameters")
        W = st.slider("Double-well height (W)", 0.1, 5.0, 1.0, 0.1,
                     help="Controls energy barrier between phases")
        kappa = st.slider("Gradient coefficient (κ)", 0.1, 10.0, 2.0, 0.1,
                         help="Controls interface width")
        M = st.slider("Mobility (M)", 0.01, 5.0, 1.0, 0.01,
                     help="Controls phase separation speed")
        sim.set_parameters(W=W, kappa=kappa, M=M)
    
    # Main FDM visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Finite-Difference Simulation")
        fig1, ax1 = plt.subplots(figsize=(8, 7))
        im1 = ax1.imshow(sim.c, cmap='bwr', origin='lower', vmin=0, vmax=1)
        ax1.set_title(f"Concentration Field (t = {sim.time:.1f})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.colorbar(im1, ax=ax1, label="Concentration c")
        st.pyplot(fig1)
        plt.close()
    
    with col2:
        st.subheader("FDM Free Energy Density")
        energy = sim.compute_free_energy_density()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(energy, cmap='viridis', origin='lower')
        ax2.set_title("Free Energy Density")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.colorbar(im2, ax=ax2, label="Energy Density")
        st.pyplot(fig2)
        plt.close()
        
        # FDM histogram
        st.subheader("FDM Concentration Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.hist(sim.c.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Concentration c")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Histogram")
        ax3.grid(True, alpha=0.3)
        st.pyplot(fig3)
        plt.close()
    
    # FDM time evolution plots
    if len(sim.history['time']) > 1:
        st.subheader("FDM Time Evolution")
        fig4, axes = plt.subplots(1, 2, figsize=(12, 4))
        
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
        
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()
    
    st.divider()
    
    # Add the PINN section
    add_pinn_section_safe()

# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
