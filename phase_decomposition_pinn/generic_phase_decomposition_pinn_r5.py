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
from typing import Dict, Tuple, Optional

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

# Default parameters
DEFAULT_PARAMS = {
    'nx': 128,
    'ny': 128,
    'Lx': 60.0,
    'Ly': 60.0,
    'T_max': 50.0,
    'W': 1.0,
    'kappa': 2.0,
    'M': 1.0,
    'c0': 0.5,
    'noise_amp': 0.05,
    'epochs': 3000,
    'lr': 1e-3,
    'pde_points': 8000,
    'ic_points': 2000,
}

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
# PINN Model Definition
# =====================================================

class CahnHilliardPINN(nn.Module):
    """Physics-Informed Neural Network for Cahn-Hilliard equation with periodic BCs."""
    
    def __init__(self, Lx: float, Ly: float, T_max: float, 
                 A: float, B: float, C: float, 
                 kappa: float, M: float, 
                 c0: float = 0.5, noise_amp: float = 0.01):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        self.A = A
        self.B = B
        self.C = C
        self.kappa = kappa
        self.M = M
        self.c0 = c0
        self.noise_amp = noise_amp
        
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
    
    c_target = model.c0 + model.noise_amp * smooth_noise
    c_pred = model(x, y, torch.zeros_like(x))
    
    return torch.mean((c_pred - c_target)**2)

# =====================================================
# PINN Training Functions
# =====================================================

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

def train_pinn_model(params: Dict, 
                    progress_callback=None,
                    status_callback=None) -> Tuple[Optional[CahnHilliardPINN], Dict]:
    """Train the PINN model with comprehensive loss tracking."""
    logger.info(f"Starting PINN training with parameters: {params}")
    
    # Extract parameters
    Lx = params['Lx']
    Ly = params['Ly']
    T_max = params['T_max']
    W = params['W']
    kappa = params['kappa']
    M = params['M']
    c0 = params['c0']
    noise_amp = params['noise_amp']
    epochs = params['epochs']
    lr = params['lr']
    pde_points = params['pde_points']
    ic_points = params['ic_points']
    
    # Set up free energy coefficients (standard double-well)
    A = W
    B = -2.0 * W
    C = W
    
    # Initialize model
    model = CahnHilliardPINN(
        Lx=Lx, Ly=Ly, T_max=T_max,
        A=A, B=B, C=C,
        kappa=kappa, M=M,
        c0=c0, noise_amp=noise_amp
    )
    
    # Generate collocation points
    points = generate_collocation_points(pde_points, ic_points, Lx, Ly, T_max)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=500 #, verbose=False
    )
    
    # Loss history tracking
    loss_history = {
        'epochs': [],
        'total': [],
        'pde': [],
        'ic': []
    }
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        try:
            # PDE loss
            residual = pde_residual(model, points['x_pde'], points['y_pde'], points['t_pde'])
            pde_loss = torch.mean(residual**2)
            
            # Initial condition loss
            ic_loss = initial_condition_loss(model, points['x_ic'], points['y_ic'])
            
            # Total loss (IC weighted higher for better constraint)
            total_loss = pde_loss + 10.0 * ic_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(total_loss)
            
        except RuntimeError as e:
            logger.error(f"Training failed at epoch {epoch}: {str(e)}")
            if "out of memory" in str(e):
                st.error("CUDA out of memory! Reduce collocation points or batch size.")
            else:
                st.error(f"Training error: {str(e)}")
            return None, {}
        
        # Log progress
        if (epoch + 1) % 100 == 0:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total'].append(total_loss.item())
            loss_history['pde'].append(pde_loss.item())
            loss_history['ic'].append(ic_loss.item())
            
            if progress_callback:
                progress_callback((epoch + 1) / epochs)
            if status_callback:
                status_callback(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Total: {total_loss.item():.6f} | "
                    f"PDE: {pde_loss.item():.6f} | "
                    f"IC: {ic_loss.item():.6f}"
                )
    
    logger.info("PINN training completed successfully")
    return model, loss_history

# =====================================================
# PINN Evaluation & Visualization
# =====================================================

def evaluate_pinn_solution(model: CahnHilliardPINN, 
                         times: np.ndarray,
                         nx: int = 128, ny: int = 128) -> Dict:
    """Evaluate the trained PINN on a regular grid for visualization."""
    Lx, Ly = model.Lx, model.Ly
    
    # Create evaluation grid
    x = torch.linspace(0, Lx, nx)
    y = torch.linspace(0, Ly, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    c_solutions = []
    free_energy_solutions = []
    
    for t_val in times:
        # Prepare time tensor
        t = torch.full((X.numel(), 1), t_val, dtype=torch.float32)
        
        # Evaluate concentration
        with torch.no_grad():
            c_pred = model(
                X.reshape(-1, 1), 
                Y.reshape(-1, 1), 
                t
            )
            c = c_pred.reshape(nx, ny).cpu().numpy()
            c_solutions.append(c)
            
            # Compute free energy density
            A, B, C = model.A, model.B, model.C
            energy = A * c**2 + B * c**3 + C * c**4
            free_energy_solutions.append(energy)
    
    return {
        'X': X.cpu().numpy(),
        'Y': Y.cpu().numpy(),
        'c_solutions': c_solutions,
        'free_energy_solutions': free_energy_solutions,
        'times': times,
        'params': {
            'Lx': Lx, 'Ly': Ly, 'T_max': model.T_max,
            'W': model.A, 'kappa': model.kappa, 'M': model.M,
            'c0': model.c0, 'noise_amp': model.noise_amp
        }
    }

def plot_pinn_losses(loss_history: Dict, output_dir: str) -> str:
    """Plot training losses."""
    plt.figure(figsize=(10, 6))
    epochs = loss_history['epochs']
    
    plt.plot(epochs, loss_history['total'], 'k-', linewidth=2, label='Total Loss')
    plt.plot(epochs, loss_history['pde'], 'b--', linewidth=1.5, label='PDE Loss')
    plt.plot(epochs, loss_history['ic'], 'r-.', linewidth=1.5, label='IC Loss')
    
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Losses')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'pinn_loss_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_pinn_concentration(solution: Dict, time_idx: int, output_dir: str) -> str:
    """Plot concentration field at specific time."""
    c = solution['c_solutions'][time_idx]
    t_val = solution['times'][time_idx]
    
    plt.figure(figsize=(8, 7))
    im = plt.imshow(c.T, cmap='bwr', origin='lower', 
                    extent=[0, solution['params']['Lx'], 0, solution['params']['Ly']],
                    vmin=0, vmax=1)
    plt.colorbar(im, label='Concentration c')
    plt.title(f'PINN Concentration Field (t = {t_val:.1f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'pinn_concentration_t{t_val:.1f}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_pinn_free_energy(solution: Dict, time_idx: int, output_dir: str) -> str:
    """Plot free energy density at specific time."""
    energy = solution['free_energy_solutions'][time_idx]
    t_val = solution['times'][time_idx]
    
    plt.figure(figsize=(6, 5))
    im = plt.imshow(energy.T, cmap='viridis', origin='lower',
                    extent=[0, solution['params']['Lx'], 0, solution['params']['Ly']])
    plt.colorbar(im, label='Free Energy Density')
    plt.title(f'PINN Free Energy Density (t = {t_val:.1f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'pinn_energy_t{t_val:.1f}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

def plot_pinn_histogram(solution: Dict, time_idx: int, output_dir: str) -> str:
    """Plot concentration histogram."""
    c = solution['c_solutions'][time_idx]
    t_val = solution['times'][time_idx]
    
    plt.figure(figsize=(6, 4))
    plt.hist(c.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlim(0, 1)
    plt.xlabel('Concentration c')
    plt.ylabel('Frequency')
    plt.title(f'Concentration Distribution (t = {t_val:.1f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'pinn_histogram_t{t_val:.1f}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_path

# =====================================================
# Streamlit PINN Integration
# =====================================================

def add_pinn_section():
    """Add PINN section to the existing Streamlit app."""
    st.header("🧠 Physics-Informed Neural Network (PINN)")
    st.markdown("""
    This section uses a **mesh-free neural network** to solve the Cahn-Hilliard equation directly from physics.
    No numerical discretization—just automatic differentiation and optimization!
    """)
    
    # PINN parameters in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("🧠 PINN Parameters")
        
        pinn_epochs = st.number_input("Training epochs", 1000, 10000, 3000, 100)
        pinn_lr = st.number_input("Learning rate", 1e-5, 1e-2, 1e-3, format="%.1e")
        pde_points = st.number_input("PDE collocation points", 1000, 20000, 8000, 1000)
        ic_points = st.number_input("IC points", 500, 5000, 2000, 500)
        
        st.info("💡 **Tip**: Start with fewer points for faster testing, then increase for accuracy.")
    
    # Get current simulation parameters
    if 'sim' in st.session_state:
        sim = st.session_state.sim
        current_params = {
            'Lx': float(sim.dx * sim.nx),
            'Ly': float(sim.dx * sim.ny),
            'T_max': DEFAULT_PARAMS['T_max'],  # Fixed for PINN
            'W': sim.W,
            'kappa': sim.kappa,
            'M': sim.M,
            'c0': 0.5,  # Fixed for consistency
            'noise_amp': 0.05,  # Fixed for consistency
            'epochs': pinn_epochs,
            'lr': pinn_lr,
            'pde_points': pde_points,
            'ic_points': ic_points,
            'nx': DEFAULT_PARAMS['nx'],
            'ny': DEFAULT_PARAMS['ny']
        }
    else:
        current_params = DEFAULT_PARAMS.copy()
        current_params.update({
            'epochs': pinn_epochs,
            'lr': pinn_lr,
            'pde_points': pde_points,
            'ic_points': ic_points
        })
    
    # PINN training button
    if st.button("🚀 Train PINN Model", type="primary"):
        with st.spinner("Training PINN... This may take several minutes."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
            
            def update_status(status):
                status_text.text(status)
            
            try:
                start_time = time.time()
                model, loss_history = train_pinn_model(
                    current_params,
                    progress_callback=update_progress,
                    status_callback=update_status
                )
                training_time = time.time() - start_time
                
                if model is None:
                    st.error("PINN training failed!")
                    return
                
                # Store results in session state
                st.session_state.pinn_model = model
                st.session_state.pinn_loss_history = loss_history
                st.session_state.pinn_params = current_params
                st.session_state.pinn_training_time = training_time
                
                st.success(f"✅ PINN training completed in {training_time:.1f} seconds!")
                
            except Exception as e:
                logger.error(f"PINN training error: {str(e)}")
                st.error(f"PINN training failed: {str(e)}")
                return
    
    # Display results if available
    if 'pinn_model' in st.session_state:
        st.subheader("📊 PINN Results")
        
        # Training info
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Time", f"{st.session_state.pinn_training_time:.1f}s")
        col2.metric("Final PDE Loss", f"{st.session_state.pinn_loss_history['pde'][-1]:.2e}")
        col3.metric("Final IC Loss", f"{st.session_state.pinn_loss_history['ic'][-1]:.2e}")
        
        # Loss plot
        loss_plot_path = plot_pinn_losses(st.session_state.pinn_loss_history, OUTPUT_DIR)
        st.image(loss_plot_path, caption="Training Losses")
        
        # Evaluate solution
        with st.spinner("Evaluating PINN solution..."):
            times_to_evaluate = np.linspace(0, current_params['T_max'], 10)
            pinn_solution = evaluate_pinn_solution(
                st.session_state.pinn_model,
                times_to_evaluate,
                nx=current_params['nx'],
                ny=current_params['ny']
            )
            st.session_state.pinn_solution = pinn_solution
        
        # Final time visualization
        final_idx = -1
        conc_plot = plot_pinn_concentration(pinn_solution, final_idx, OUTPUT_DIR)
        energy_plot = plot_pinn_free_energy(pinn_solution, final_idx, OUTPUT_DIR)
        hist_plot = plot_pinn_histogram(pinn_solution, final_idx, OUTPUT_DIR)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(conc_plot, caption="Concentration Field")
        with col2:
            st.image(energy_plot, caption="Free Energy Density")
        
        st.image(hist_plot, caption="Concentration Distribution")
        
        # Time series animation
        st.subheader("🎬 Time Evolution")
        time_slider = st.slider(
            "Select time step", 
            0, len(times_to_evaluate)-1, 
            len(times_to_evaluate)-1,
            format="t = %.1f"
        )
        
        if time_slider != final_idx:
            conc_plot_ts = plot_pinn_concentration(pinn_solution, time_slider, OUTPUT_DIR)
            energy_plot_ts = plot_pinn_free_energy(pinn_solution, time_slider, OUTPUT_DIR)
            hist_plot_ts = plot_pinn_histogram(pinn_solution, time_slider, OUTPUT_DIR)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(conc_plot_ts, caption=f"Concentration (t = {times_to_evaluate[time_slider]:.1f})")
            with col2:
                st.image(energy_plot_ts, caption=f"Free Energy (t = {times_to_evaluate[time_slider]:.1f})")
            st.image(hist_plot_ts, caption=f"Concentration Distribution (t = {times_to_evaluate[time_slider]:.1f})")
        
        # Download section
        st.subheader("💾 Download PINN Results")
        
        # Save model
        model_path = os.path.join(OUTPUT_DIR, 'pinn_model.pth')
        torch.save(st.session_state.pinn_model.state_dict(), model_path)
        
        with open(model_path, 'rb') as f:
            st.download_button(
                "📥 Download Trained Model",
                f.read(),
                "pinn_model.pth",
                "application/octet-stream"
            )
        
        # Save solution
        solution_path = os.path.join(OUTPUT_DIR, 'pinn_solution.pkl')
        with open(solution_path, 'wb') as f:
            pickle.dump(pinn_solution, f)
        
        with open(solution_path, 'rb') as f:
            st.download_button(
                "📥 Download Solution Data",
                f.read(),
                "pinn_solution.pkl",
                "application/octet-stream"
            )
        
        # Save loss history
        loss_history_path = os.path.join(OUTPUT_DIR, 'pinn_loss_history.pkl')
        with open(loss_history_path, 'wb') as f:
            pickle.dump(st.session_state.pinn_loss_history, f)
        
        with open(loss_history_path, 'rb') as f:
            st.download_button(
                "📥 Download Loss History",
                f.read(),
                "pinn_loss_history.pkl",
                "application/octet-stream"
            )

# =====================================================
# Original Streamlit App (with PINN integration)
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
    
    # ==============================
    # ADD PINN SECTION HERE
    # ==============================
    st.divider()
    add_pinn_section()

if __name__ == "__main__":
    main()
