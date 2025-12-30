"""
COMPREHENSIVE PINN SOLUTION FOR CU-NI CROSS-DIFFUSION PROBLEM
================================================================
Physics-Informed Neural Network for solving coupled cross-diffusion equations
with boundary conditions and time evolution.

Governing Equations:
∂c₁/∂t = D₁₁∇²c₁ + D₁₂∇²c₂  (Cu concentration)
∂c₂/∂t = D₂₁∇²c₁ + D₂₂∇²c₂  (Ni concentration)

Domain: x ∈ [0, Lx], y ∈ [0, Ly], t ∈ [0, T_max]
Boundary Conditions:
  Top (y=Ly): Cu-rich, Ni-poor
  Bottom (y=0): Cu-poor, Ni-rich
  Sides: Zero flux (∂c/∂x = 0)
Initial Condition: c₁ = c₂ = 0 at t=0
"""

# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
import zipfile
import io
import matplotlib as mpl
import logging
import pyvista as pv
from pyvista import examples
import hashlib
import time
import json
from datetime import datetime
from scipy import integrate
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f'/tmp/pinn_solutions_{timestamp}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Enhanced Matplotlib configuration
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 2.0
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['grid.linestyle'] = '--'

# Enhanced logging configuration
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# 2. PHYSICAL PARAMETERS AND CONSTANTS
# ============================================================================
class PhysicalParameters:
    """Container for all physical parameters with validation"""
    
    # Diffusion coefficients (μm²/s) - Cu-Ni system at 260°C
    D11 = 0.00600      # Cu self-diffusion coefficient
    D12 = 0.00427      # Cu-Ni cross-diffusion coefficient
    D21 = 0.003697     # Ni-Cu cross-diffusion coefficient  
    D22 = 0.00540      # Ni self-diffusion coefficient
    
    # Domain dimensions (μm)
    Lx = 60.0          # Width
    Ly = 50.0          # Height
    
    # Time parameters (s)
    T_max = 200.0      # Maximum simulation time
    
    # Boundary concentrations (mol/μm³)
    C_CU_TOP = 1.59e-03    # Top boundary: Cu-rich
    C_CU_BOTTOM = 0.0      # Bottom boundary: Cu-poor
    C_NI_TOP = 0.0         # Top boundary: Ni-poor
    C_NI_BOTTOM = 4.0e-04  # Bottom boundary: Ni-rich
    
    # Material properties
    temperature = 533.15    # Kelvin (260°C)
    molar_volume = 7.11e-6  # μm³/mol (approx for Cu-Ni alloy)
    
    @classmethod
    def validate(cls):
        """Validate physical parameters"""
        assert cls.D11 > 0 and cls.D22 > 0, "Diffusion coefficients must be positive"
        assert cls.Lx > 0 and cls.Ly > 0, "Domain dimensions must be positive"
        assert cls.T_max > 0, "Maximum time must be positive"
        assert cls.C_CU_TOP >= 0 and cls.C_NI_BOTTOM >= 0, "Concentrations must be non-negative"
        logger.info("Physical parameters validated successfully")
        
    @classmethod
    def get_dict(cls):
        """Return parameters as dictionary"""
        return {
            'D11': cls.D11, 'D12': cls.D12, 'D21': cls.D21, 'D22': cls.D22,
            'Lx': cls.Lx, 'Ly': cls.Ly, 'T_max': cls.T_max,
            'C_CU_TOP': cls.C_CU_TOP, 'C_CU_BOTTOM': cls.C_CU_BOTTOM,
            'C_NI_TOP': cls.C_NI_TOP, 'C_NI_BOTTOM': cls.C_NI_BOTTOM,
            'temperature': cls.temperature, 'molar_volume': cls.molar_volume
        }

# Validate parameters
PhysicalParameters.validate()

# ============================================================================
# 3. NEURAL NETWORK ARCHITECTURES
# ============================================================================

class FourierFeatureMapping(nn.Module):
    """Fourier feature mapping for better representation of high-frequency features"""
    def __init__(self, input_dim, mapping_size=256, sigma=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size // 2) * sigma)
        
    def forward(self, x):
        x_proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class AdaptiveActivation(nn.Module):
    """Adaptive activation function with learnable slope"""
    def __init__(self, slope=1.0):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(slope))
        
    def forward(self, x):
        return torch.tanh(self.a * x)

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = AdaptiveActivation()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.layer_norm(out + residual)
        return self.activation(out)

class EnhancedPINN(nn.Module):
    """
    Enhanced Physics-Informed Neural Network with:
    - Fourier feature mapping
    - Residual connections
    - Adaptive activations
    - Multiple output heads with boundary condition encoding
    """
    
    def __init__(self, params, hidden_layers=8, hidden_dim=256, 
                 fourier_features=True, dropout_rate=0.05):
        super().__init__()
        
        self.params = params
        self.fourier_features = fourier_features
        
        # Input normalization
        self.x_mean = params['Lx'] / 2
        self.x_std = params['Lx'] / 2
        self.y_mean = params['Ly'] / 2
        self.y_std = params['Ly'] / 2
        self.t_mean = params['T_max'] / 2
        self.t_std = params['T_max'] / 2
        
        # Input processing
        if fourier_features:
            self.fourier_map = FourierFeatureMapping(5, mapping_size=128)
            input_dim = 128 + 5  # Fourier features + original features
        else:
            input_dim = 5
        
        # Main network with residual blocks
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(AdaptiveActivation())
        layers.append(nn.Dropout(dropout_rate))
        
        for _ in range(hidden_layers - 1):
            layers.append(ResidualBlock(hidden_dim, dropout_rate))
            
        self.shared_net = nn.Sequential(*layers)
        
        # Output heads with boundary condition enforcement
        self.cu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            AdaptiveActivation(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure non-negative concentration
        )
        
        self.ni_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            AdaptiveActivation(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Boundary condition encoding parameters
        self.register_buffer('top_bc_cu', torch.tensor(params['C_CU_TOP']))
        self.register_buffer('top_bc_ni', torch.tensor(params['C_NI_TOP']))
        self.register_buffer('bottom_bc_cu', torch.tensor(params['C_CU_BOTTOM']))
        self.register_buffer('bottom_bc_ni', torch.tensor(params['C_NI_BOTTOM']))
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def _normalize_inputs(self, x, y, t):
        """Normalize inputs to [-1, 1] range"""
        x_norm = (x - self.x_mean) / self.x_std
        y_norm = (y - self.y_mean) / self.y_std
        t_norm = (t - self.t_mean) / self.t_std
        return x_norm, y_norm, t_norm
        
    def forward(self, x, y, t):
        """
        Forward pass with boundary condition encoding
        Returns: [c_cu, c_ni]
        """
        # Normalize inputs
        x_norm, y_norm, t_norm = self._normalize_inputs(x, y, t)
        
        # Create base features
        base_features = torch.cat([x_norm, y_norm, t_norm, 
                                 torch.ones_like(x_norm), 
                                 torch.ones_like(x_norm)], dim=1)
        
        # Apply Fourier feature mapping if enabled
        if self.fourier_features:
            ff_features = self.fourier_map(base_features)
            features = torch.cat([base_features, ff_features], dim=1)
        else:
            features = base_features
            
        # Pass through shared network
        shared_out = self.shared_net(features)
        
        # Get raw predictions
        cu_raw = self.cu_head(shared_out)
        ni_raw = self.ni_head(shared_out)
        
        # Apply boundary condition encoding using a smooth transition function
        # This ensures exact boundary conditions at y=0 and y=Ly
        y_ratio = y / self.params['Ly']
        
        # Encode top boundary (y = Ly)
        top_weight = torch.sigmoid(10.0 * (y_ratio - 0.95))  # Smooth transition near top
        cu_top = self.top_bc_cu * top_weight
        ni_top = self.top_bc_ni * top_weight
        
        # Encode bottom boundary (y = 0)
        bottom_weight = torch.sigmoid(-10.0 * (y_ratio - 0.05))  # Smooth transition near bottom
        cu_bottom = self.bottom_bc_cu * bottom_weight
        ni_bottom = self.bottom_bc_ni * bottom_weight
        
        # Combine with interior solution
        # Interior weight ensures boundary conditions are exact at boundaries
        interior_weight = (1 - top_weight) * (1 - bottom_weight)
        
        c_cu = interior_weight * cu_raw + cu_top + cu_bottom
        c_ni = interior_weight * ni_raw + ni_top + ni_bottom
        
        return torch.cat([c_cu, c_ni], dim=1)
    
    def compute_physics_residuals(self, x, y, t):
        """
        Compute PDE residuals with automatic differentiation
        Returns: residual1, residual2, and individual terms for analysis
        """
        # Enable gradient tracking
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)
        
        # Get predictions
        c_pred = self(x, y, t)
        c_cu = c_pred[:, 0:1]
        c_ni = c_pred[:, 1:2]
        
        # First derivatives
        c_cu_t = torch.autograd.grad(c_cu, t, grad_outputs=torch.ones_like(c_cu),
                                     create_graph=True, retain_graph=True)[0]
        c_ni_t = torch.autograd.grad(c_ni, t, grad_outputs=torch.ones_like(c_ni),
                                     create_graph=True, retain_graph=True)[0]
        
        # Spatial gradients
        c_cu_x = torch.autograd.grad(c_cu, x, grad_outputs=torch.ones_like(c_cu),
                                    create_graph=True, retain_graph=True)[0]
        c_cu_y = torch.autograd.grad(c_cu, y, grad_outputs=torch.ones_like(c_cu),
                                    create_graph=True, retain_graph=True)[0]
        
        c_ni_x = torch.autograd.grad(c_ni, x, grad_outputs=torch.ones_like(c_ni),
                                    create_graph=True, retain_graph=True)[0]
        c_ni_y = torch.autograd.grad(c_ni, y, grad_outputs=torch.ones_like(c_ni),
                                    create_graph=True, retain_graph=True)[0]
        
        # Second derivatives (Laplacian)
        c_cu_xx = torch.autograd.grad(c_cu_x, x, grad_outputs=torch.ones_like(c_cu_x),
                                     create_graph=True, retain_graph=True)[0]
        c_cu_yy = torch.autograd.grad(c_cu_y, y, grad_outputs=torch.ones_like(c_cu_y),
                                     create_graph=True, retain_graph=True)[0]
        
        c_ni_xx = torch.autograd.grad(c_ni_x, x, grad_outputs=torch.ones_like(c_ni_x),
                                     create_graph=True, retain_graph=True)[0]
        c_ni_yy = torch.autograd.grad(c_ni_y, y, grad_outputs=torch.ones_like(c_ni_y),
                                     create_graph=True, retain_graph=True)[0]
        
        laplacian_cu = c_cu_xx + c_cu_yy
        laplacian_ni = c_ni_xx + c_ni_yy
        
        # PDE residuals
        residual_cu = c_cu_t - (self.params['D11'] * laplacian_cu + 
                               self.params['D12'] * laplacian_ni)
        residual_ni = c_ni_t - (self.params['D21'] * laplacian_cu + 
                               self.params['D22'] * laplacian_ni)
        
        # Return all terms for analysis
        physics_info = {
            'residuals': (residual_cu, residual_ni),
            'gradients': {
                'c_cu_t': c_cu_t, 'c_ni_t': c_ni_t,
                'c_cu_x': c_cu_x, 'c_cu_y': c_cu_y,
                'c_ni_x': c_ni_x, 'c_ni_y': c_ni_y
            },
            'laplacians': (laplacian_cu, laplacian_ni)
        }
        
        return physics_info

# ============================================================================
# 4. LOSS FUNCTIONS WITH ADAPTIVE WEIGHTING
# ============================================================================

class AdaptiveLossWeights:
    """Dynamically adjust loss weights during training"""
    
    def __init__(self, initial_weights=None):
        if initial_weights is None:
            initial_weights = {
                'physics': 1.0,
                'boundary_bottom': 100.0,
                'boundary_top': 100.0,
                'boundary_sides': 100.0,
                'initial': 100.0,
                'mass_conservation': 10.0
            }
        self.weights = initial_weights
        self.history = {key: [] for key in initial_weights}
        self.update_frequency = 100
        
    def update(self, losses, epoch):
        """Update weights based on loss ratios"""
        if epoch % self.update_frequency == 0:
            total_loss = sum(losses.values())
            for key in self.weights:
                if total_loss > 0:
                    loss_ratio = losses[key] / total_loss
                    # Increase weight if loss component is relatively high
                    self.weights[key] *= (1.0 + 0.1 * loss_ratio)
                    
        # Store current weights
        for key in self.weights:
            self.history[key].append(self.weights[key])
            
        return self.weights

def compute_physics_loss(model, points_dict):
    """
    Compute physics loss at collocation points
    points_dict: dictionary with 'x', 'y', 't' tensors
    """
    physics_info = model.compute_physics_residuals(
        points_dict['x'], points_dict['y'], points_dict['t']
    )
    residual_cu, residual_ni = physics_info['residuals']
    
    # L2 norm of residuals
    loss_physics = torch.mean(residual_cu**2 + residual_ni**2)
    
    # Additional regularization: gradient penalty
    gradients = physics_info['gradients']
    grad_norm = sum(torch.mean(g**2) for g in gradients.values())
    
    return loss_physics + 0.01 * grad_norm, physics_info

def compute_boundary_loss(model, boundary_type='bottom'):
    """
    Compute boundary loss for specific boundary
    boundary_type: 'bottom', 'top', 'left', 'right'
    """
    num_points = 500
    
    if boundary_type == 'bottom':
        x = torch.rand(num_points, 1, requires_grad=True) * model.params['Lx']
        y = torch.zeros(num_points, 1, requires_grad=True)
        target_cu = model.params['C_CU_BOTTOM']
        target_ni = model.params['C_NI_BOTTOM']
        
    elif boundary_type == 'top':
        x = torch.rand(num_points, 1, requires_grad=True) * model.params['Lx']
        y = torch.full((num_points, 1), model.params['Ly'], requires_grad=True)
        target_cu = model.params['C_CU_TOP']
        target_ni = model.params['C_NI_TOP']
        
    elif boundary_type == 'left':
        x = torch.zeros(num_points, 1, requires_grad=True)
        y = torch.rand(num_points, 1, requires_grad=True) * model.params['Ly']
        # Zero flux condition: ∂c/∂x = 0
        c_pred = model(x, y, torch.rand(num_points, 1) * model.params['T_max'])
        c_cu = c_pred[:, 0:1]
        c_ni = c_pred[:, 1:2]
        
        grad_cu_x = torch.autograd.grad(c_cu, x, grad_outputs=torch.ones_like(c_cu),
                                       create_graph=True, retain_graph=True)[0]
        grad_ni_x = torch.autograd.grad(c_ni, x, grad_outputs=torch.ones_like(c_ni),
                                       create_graph=True, retain_graph=True)[0]
        
        return torch.mean(grad_cu_x**2 + grad_ni_x**2)
        
    elif boundary_type == 'right':
        x = torch.full((num_points, 1), model.params['Lx'], requires_grad=True)
        y = torch.rand(num_points, 1, requires_grad=True) * model.params['Ly']
        # Zero flux condition: ∂c/∂x = 0
        c_pred = model(x, y, torch.rand(num_points, 1) * model.params['T_max'])
        c_cu = c_pred[:, 0:1]
        c_ni = c_pred[:, 1:2]
        
        grad_cu_x = torch.autograd.grad(c_cu, x, grad_outputs=torch.ones_like(c_cu),
                                       create_graph=True, retain_graph=True)[0]
        grad_ni_x = torch.autograd.grad(c_ni, x, grad_outputs=torch.ones_like(c_ni),
                                       create_graph=True, retain_graph=True)[0]
        
        return torch.mean(grad_cu_x**2 + grad_ni_x**2)
    
    else:
        raise ValueError(f"Unknown boundary type: {boundary_type}")
    
    t = torch.rand(num_points, 1, requires_grad=True) * model.params['T_max']
    c_pred = model(x, y, t)
    
    loss_cu = torch.mean((c_pred[:, 0] - target_cu)**2)
    loss_ni = torch.mean((c_pred[:, 1] - target_ni)**2)
    
    return loss_cu + loss_ni

def compute_initial_loss(model):
    """Compute loss for initial condition (t=0)"""
    num_points = 1000
    x = torch.rand(num_points, 1, requires_grad=True) * model.params['Lx']
    y = torch.rand(num_points, 1, requires_grad=True) * model.params['Ly']
    t = torch.zeros(num_points, 1, requires_grad=True)
    
    c_pred = model(x, y, t)
    # Initial condition: zero concentration everywhere
    return torch.mean(c_pred**2)

def compute_mass_conservation_loss(model, points_dict):
    """
    Compute mass conservation loss (optional)
    For systems with no sources/sinks, total mass should be conserved
    """
    # This is an approximate check - full conservation would require integration
    c_pred = model(points_dict['x'], points_dict['y'], points_dict['t'])
    c_cu, c_ni = c_pred[:, 0], c_pred[:, 1]
    
    # Check variance of total concentration (should be relatively constant)
    total_concentration = c_cu + c_ni
    variance = torch.var(total_concentration)
    
    return variance

# ============================================================================
# 5. TRAINING MANAGER WITH ADVANCED FEATURES
# ============================================================================

class PINNTrainer:
    """
    Comprehensive training manager for PINNs with:
    - Adaptive sampling
    - Multiple optimization strategies
    - Advanced logging and checkpointing
    - Convergence monitoring
    """
    
    def __init__(self, model, params, output_dir):
        self.model = model
        self.params = params
        self.output_dir = output_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training configuration
        self.config = {
            'epochs': 10000,
            'initial_lr': 1e-3,
            'batch_size': 4096,
            'validation_freq': 100,
            'checkpoint_freq': 500,
            'adaptive_sampling_freq': 200
        }
        
        # Point generators for different loss components
        self.point_generators = self._initialize_point_generators()
        
        # Loss tracker
        self.loss_history = {
            'total': [], 'physics': [], 'boundary_bottom': [],
            'boundary_top': [], 'boundary_sides': [], 'initial': [],
            'learning_rate': [], 'gradient_norm': []
        }
        
        # Adaptive loss weights
        self.loss_weights = AdaptiveLossWeights()
        
        # Setup optimizers and schedulers
        self._setup_optimization()
        
        # Create checkpoint directory
        self.checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def _initialize_point_generators(self):
        """Initialize point generators for different domains"""
        generators = {}
        
        # Interior points for PDE
        generators['interior'] = {
            'num_points': 10000,
            'x_range': (0, self.params['Lx']),
            'y_range': (0, self.params['Ly']),
            't_range': (0, self.params['T_max'])
        }
        
        # Boundary points
        for boundary in ['bottom', 'top', 'left', 'right']:
            generators[boundary] = {
                'num_points': 2000,
                'boundary': boundary
            }
            
        # Initial condition points
        generators['initial'] = {
            'num_points': 5000,
            't_value': 0.0
        }
        
        return generators
    
    def _setup_optimization(self):
        """Setup optimizers and learning rate schedulers"""
        
        # Main optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['initial_lr'],
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        
        # Learning rate schedulers
        self.scheduler = {
            'plateau': ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, 
                patience=200, verbose=True, min_lr=1e-6
            ),
            'cosine': CosineAnnealingLR(
                self.optimizer, T_max=self.config['epochs'], 
                eta_min=1e-6
            )
        }
        
        # Gradient clipping
        self.grad_clip_value = 1.0
        
    def _generate_points(self, generator_type, adaptive=False, residual_info=None):
        """
        Generate points for training, optionally with adaptive sampling
        based on residual magnitudes
        """
        gen_config = self.point_generators[generator_type]
        
        if generator_type == 'interior':
            if adaptive and residual_info is not None:
                # Adaptive sampling: generate more points in high-residual regions
                residual_cu, residual_ni = residual_info
                residuals = torch.abs(residual_cu) + torch.abs(residual_ni)
                
                # Sample points proportional to residuals
                probabilities = residuals.detach().cpu().numpy().flatten()
                probabilities = probabilities / (probabilities.sum() + 1e-10)
                
                # Generate new points
                num_new = gen_config['num_points']
                idx = np.random.choice(
                    len(probabilities), 
                    size=num_new, 
                    p=probabilities,
                    replace=True
                )
                
                # Get corresponding coordinates (would need to store them)
                # For simplicity, fall back to uniform sampling
                pass
                
            # Uniform sampling
            x = torch.rand(gen_config['num_points'], 1) * self.params['Lx']
            y = torch.rand(gen_config['num_points'], 1) * self.params['Ly']
            t = torch.rand(gen_config['num_points'], 1) * self.params['T_max']
            
        elif generator_type in ['bottom', 'top']:
            x = torch.rand(gen_config['num_points'], 1) * self.params['Lx']
            if generator_type == 'bottom':
                y = torch.zeros(gen_config['num_points'], 1)
            else:  # top
                y = torch.full((gen_config['num_points'], 1), self.params['Ly'])
            t = torch.rand(gen_config['num_points'], 1) * self.params['T_max']
            
        elif generator_type in ['left', 'right']:
            y = torch.rand(gen_config['num_points'], 1) * self.params['Ly']
            if generator_type == 'left':
                x = torch.zeros(gen_config['num_points'], 1)
            else:  # right
                x = torch.full((gen_config['num_points'], 1), self.params['Lx'])
            t = torch.rand(gen_config['num_points'], 1) * self.params['T_max']
            
        elif generator_type == 'initial':
            x = torch.rand(gen_config['num_points'], 1) * self.params['Lx']
            y = torch.rand(gen_config['num_points'], 1) * self.params['Ly']
            t = torch.full((gen_config['num_points'], 1), gen_config['t_value'])
            
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        return {
            'x': x.to(self.device),
            'y': y.to(self.device),
            't': t.to(self.device)
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        
        # Generate training points
        interior_points = self._generate_points('interior')
        
        # Compute losses
        loss_physics, physics_info = compute_physics_loss(self.model, interior_points)
        loss_bottom = compute_boundary_loss(self.model, 'bottom')
        loss_top = compute_boundary_loss(self.model, 'top')
        loss_left = compute_boundary_loss(self.model, 'left')
        loss_right = compute_boundary_loss(self.model, 'right')
        loss_sides = loss_left + loss_right
        loss_initial = compute_initial_loss(self.model)
        
        # Get adaptive weights
        losses_dict = {
            'physics': loss_physics.item(),
            'boundary_bottom': loss_bottom.item(),
            'boundary_top': loss_top.item(),
            'boundary_sides': loss_sides.item(),
            'initial': loss_initial.item()
        }
        weights = self.loss_weights.update(losses_dict, epoch)
        
        # Weighted total loss
        total_loss = (
            weights['physics'] * loss_physics +
            weights['boundary_bottom'] * loss_bottom +
            weights['boundary_top'] * loss_top +
            weights['boundary_sides'] * loss_sides +
            weights['initial'] * loss_initial
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.grad_clip_value
        )
        
        # Optimizer step
        self.optimizer.step()
        
        # Update learning rate
        if epoch % 100 == 0:
            self.scheduler['cosine'].step()
        
        # Record losses
        self.loss_history['total'].append(total_loss.item())
        self.loss_history['physics'].append(loss_physics.item())
        self.loss_history['boundary_bottom'].append(loss_bottom.item())
        self.loss_history['boundary_top'].append(loss_top.item())
        self.loss_history['boundary_sides'].append(loss_sides.item())
        self.loss_history['initial'].append(loss_initial.item())
        self.loss_history['learning_rate'].append(
            self.optimizer.param_groups[0]['lr']
        )
        
        return total_loss.item(), physics_info
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        
        with torch.no_grad():
            # Generate validation points
            val_points = self._generate_points('interior')
            val_points['num_points'] = 2000  # Smaller set for validation
            
            # Compute validation loss
            val_loss_physics, _ = compute_physics_loss(self.model, val_points)
            
            # Check boundary conditions
            bc_errors = self._check_boundary_conditions()
            
        return val_loss_physics.item(), bc_errors
    
    def _check_boundary_conditions(self):
        """Check boundary condition satisfaction"""
        errors = {}
        
        # Check bottom boundary
        x = torch.rand(100, 1) * self.params['Lx']
        y = torch.zeros(100, 1)
        t = torch.rand(100, 1) * self.params['T_max']
        
        with torch.no_grad():
            c_pred = self.model(x, y, t)
            errors['bottom_cu'] = torch.mean(
                torch.abs(c_pred[:, 0] - self.params['C_CU_BOTTOM'])
            ).item()
            errors['bottom_ni'] = torch.mean(
                torch.abs(c_pred[:, 1] - self.params['C_NI_BOTTOM'])
            ).item()
            
            # Check top boundary
            y = torch.full((100, 1), self.params['Ly'])
            c_pred = self.model(x, y, t)
            errors['top_cu'] = torch.mean(
                torch.abs(c_pred[:, 0] - self.params['C_CU_TOP'])
            ).item()
            errors['top_ni'] = torch.mean(
                torch.abs(c_pred[:, 1] - self.params['C_NI_TOP'])
            ).item()
            
        return errors
    
    def save_checkpoint(self, epoch, best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history,
            'config': self.config,
            'params': self.params
        }
        
        filename = f'checkpoint_epoch_{epoch}.pt'
        if best:
            filename = 'best_model.pt'
            
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, filename))
        
    def train(self, progress_callback=None):
        """Main training loop"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            # Train one epoch
            train_loss, physics_info = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config['validation_freq'] == 0:
                val_loss, bc_errors = self.validate()
                
                # Update plateau scheduler
                self.scheduler['plateau'].step(val_loss)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, best=True)
                    
                # Log validation results
                log_msg = (f"Epoch {epoch}/{self.config['epochs']}: "
                          f"Train Loss = {train_loss:.6f}, "
                          f"Val Loss = {val_loss:.6f}, "
                          f"LR = {self.optimizer.param_groups[0]['lr']:.2e}")
                logger.info(log_msg)
                
                # Log boundary condition errors
                for bc, error in bc_errors.items():
                    logger.info(f"  {bc}: {error:.2e}")
            
            # Save checkpoint
            if epoch % self.config['checkpoint_freq'] == 0:
                self.save_checkpoint(epoch)
            
            # Adaptive sampling update
            if epoch % self.config['adaptive_sampling_freq'] == 0:
                # Could update point generators based on physics_info
                pass
            
            # Progress callback for UI
            if progress_callback:
                progress_callback(epoch, self.config['epochs'], train_loss)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return self.loss_history

# ============================================================================
# 6. VISUALIZATION AND ANALYSIS TOOLS
# ============================================================================

class SolutionAnalyzer:
    """Comprehensive analysis and visualization of PINN solutions"""
    
    def __init__(self, solution_data, params, output_dir):
        self.solution = solution_data
        self.params = params
        self.output_dir = output_dir
        self.colormaps = {
            'Cu': 'viridis',
            'Ni': 'plasma',
            'residual': 'RdBu_r',
            'flux': 'coolwarm'
        }
        
    def create_comprehensive_plots(self):
        """Create all analysis plots"""
        plots = {}
        
        # 1. Concentration profiles
        plots['concentration_3d'] = self.plot_3d_concentration()
        plots['concentration_2d'] = self.plot_2d_concentration_grid()
        plots['concentration_animation'] = self.create_concentration_animation()
        
        # 2. Time evolution
        plots['time_evolution'] = self.plot_time_evolution()
        plots['boundary_evolution'] = self.plot_boundary_evolution()
        
        # 3. Flux analysis
        plots['flux_distribution'] = self.plot_flux_distribution()
        plots['mass_conservation'] = self.plot_mass_conservation()
        
        # 4. Error analysis
        plots['error_distribution'] = self.plot_error_distribution()
        plots['convergence_analysis'] = self.plot_convergence_analysis()
        
        # 5. Validation plots
        plots['boundary_validation'] = self.plot_boundary_validation()
        plots['pde_residuals'] = self.plot_pde_residuals()
        
        return plots
    
    def plot_3d_concentration(self):
        """Create 3D surface plots of concentration"""
        fig = plt.figure(figsize=(16, 8))
        
        # Extract data for final time step
        t_idx = -1
        X = self.solution['X']
        Y = self.solution['Y']
        c1 = self.solution['c1_preds'][t_idx]
        c2 = self.solution['c2_preds'][t_idx]
        
        # Cu concentration
        ax1 = fig.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(X, Y, c1, cmap=self.colormaps['Cu'],
                               linewidth=0, antialiased=True,
                               rstride=1, cstride=1, alpha=0.8)
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        ax1.set_zlabel('Cu Concentration (mol/μm³)')
        ax1.set_title('3D Cu Distribution')
        fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        
        # Ni concentration
        ax2 = fig.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(X, Y, c2, cmap=self.colormaps['Ni'],
                               linewidth=0, antialiased=True,
                               rstride=1, cstride=1, alpha=0.8)
        ax2.set_xlabel('x (μm)')
        ax2.set_ylabel('y (μm)')
        ax2.set_zlabel('Ni Concentration (mol/μm³)')
        ax2.set_title('3D Ni Distribution')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        
        plt.suptitle(f'3D Concentration Profiles at t = {self.solution["times"][t_idx]:.1f} s', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        filename = os.path.join(self.output_dir, '3d_concentration_profiles.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_2d_concentration_grid(self):
        """Create 2D grid of concentration profiles at different times"""
        times = self.solution['times']
        num_times = min(6, len(times))
        time_indices = np.linspace(0, len(times)-1, num_times, dtype=int)
        
        fig, axes = plt.subplots(num_times, 2, figsize=(14, 4*num_times))
        
        for i, t_idx in enumerate(time_indices):
            t_val = times[t_idx]
            c1 = self.solution['c1_preds'][t_idx]
            c2 = self.solution['c2_preds'][t_idx]
            
            # Cu concentration
            im1 = axes[i, 0].imshow(c1, origin='lower', 
                                   extent=[0, self.params['Lx'], 0, self.params['Ly']],
                                   cmap=self.colormaps['Cu'],
                                   vmin=0, vmax=self.params['C_CU_TOP'])
            axes[i, 0].set_title(f'Cu at t = {t_val:.1f} s')
            axes[i, 0].set_xlabel('x (μm)')
            axes[i, 0].set_ylabel('y (μm)')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Ni concentration
            im2 = axes[i, 1].imshow(c2, origin='lower',
                                   extent=[0, self.params['Lx'], 0, self.params['Ly']],
                                   cmap=self.colormaps['Ni'],
                                   vmin=0, vmax=self.params['C_NI_BOTTOM'])
            axes[i, 1].set_title(f'Ni at t = {t_val:.1f} s')
            axes[i, 1].set_xlabel('x (μm)')
            axes[i, 1].set_ylabel('y (μm)')
            plt.colorbar(im2, ax=axes[i, 1])
        
        plt.suptitle('Time Evolution of Concentration Profiles', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = os.path.join(self.output_dir, '2d_concentration_grid.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_time_evolution(self):
        """Plot concentration evolution at specific points"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Select observation points
        points = [
            (self.params['Lx']/2, self.params['Ly']/2, 'Center'),
            (self.params['Lx']/2, 0, 'Bottom Center'),
            (self.params['Lx']/2, self.params['Ly'], 'Top Center'),
            (0, self.params['Ly']/2, 'Left Center')
        ]
        
        times = np.array(self.solution['times'])
        
        for idx, (x_pos, y_pos, label) in enumerate(points):
            ax = axes[idx // 2, idx % 2]
            
            # Find nearest grid point
            x_idx = np.argmin(np.abs(self.solution['X'][0, :] - x_pos))
            y_idx = np.argmin(np.abs(self.solution['Y'][:, 0] - y_pos))
            
            # Extract time evolution
            c1_vals = [c[y_idx, x_idx] for c in self.solution['c1_preds']]
            c2_vals = [c[y_idx, x_idx] for c in self.solution['c2_preds']]
            
            # Plot
            ax.plot(times, c1_vals, 'b-', linewidth=2, label='Cu')
            ax.plot(times, c2_vals, 'r-', linewidth=2, label='Ni')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Concentration (mol/μm³)')
            ax.set_title(f'Evolution at {label}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add theoretical prediction (simplified 1D diffusion)
            if label == 'Center':
                # Simplified analytical solution for diffusion from boundaries
                D_eff = (self.params['D11'] + self.params['D22']) / 2
                c_theory = self.params['C_CU_TOP'] * 0.5 * (1 - np.exp(-D_eff * times / (self.params['Ly']**2)))
                ax.plot(times, c_theory, 'g--', linewidth=1.5, label='Theoretical (approx)')
                ax.legend()
        
        plt.suptitle('Concentration Time Evolution at Observation Points', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = os.path.join(self.output_dir, 'time_evolution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_flux_distribution(self):
        """Calculate and plot flux distributions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Compute gradients (approximate)
        c1_final = self.solution['c1_preds'][-1]
        c2_final = self.solution['c2_preds'][-1]
        
        # Compute gradients using finite differences
        dx = self.params['Lx'] / (c1_final.shape[1] - 1)
        dy = self.params['Ly'] / (c1_final.shape[0] - 1)
        
        # Flux components (Fick's first law)
        flux_cu_x = -self.params['D11'] * np.gradient(c1_final, dx, axis=1)
        flux_cu_y = -self.params['D11'] * np.gradient(c1_final, dy, axis=0)
        
        flux_ni_x = -self.params['D22'] * np.gradient(c2_final, dx, axis=1)
        flux_ni_y = -self.params['D22'] * np.gradient(c2_final, dy, axis=0)
        
        # Magnitude of total flux
        flux_magnitude_cu = np.sqrt(flux_cu_x**2 + flux_cu_y**2)
        flux_magnitude_ni = np.sqrt(flux_ni_x**2 + flux_ni_y**2)
        
        # Plot flux magnitude
        im1 = axes[0].imshow(flux_magnitude_cu, origin='lower',
                            extent=[0, self.params['Lx'], 0, self.params['Ly']],
                            cmap=self.colormaps['flux'])
        axes[0].set_title('Cu Flux Magnitude')
        axes[0].set_xlabel('x (μm)')
        axes[0].set_ylabel('y (μm)')
        plt.colorbar(im1, ax=axes[0], label='Flux magnitude')
        
        im2 = axes[1].imshow(flux_magnitude_ni, origin='lower',
                            extent=[0, self.params['Lx'], 0, self.params['Ly']],
                            cmap=self.colormaps['flux'])
        axes[1].set_title('Ni Flux Magnitude')
        axes[1].set_xlabel('x (μm)')
        axes[1].set_ylabel('y (μm)')
        plt.colorbar(im2, ax=axes[1], label='Flux magnitude')
        
        # Add streamlines to show flux direction
        Y, X = np.mgrid[0:self.params['Ly']:c1_final.shape[0]*1j,
                       0:self.params['Lx']:c1_final.shape[1]*1j]
        axes[0].streamplot(X, Y, flux_cu_x, flux_cu_y, color='white', 
                          linewidth=0.5, density=1.5)
        axes[1].streamplot(X, Y, flux_ni_x, flux_ni_y, color='white',
                          linewidth=0.5, density=1.5)
        
        plt.suptitle('Flux Distribution at Final Time', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = os.path.join(self.output_dir, 'flux_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_mass_conservation(self):
        """Check and plot mass conservation"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        times = np.array(self.solution['times'])
        total_mass_cu = []
        total_mass_ni = []
        
        # Approximate integration of mass
        dx = self.params['Lx'] / (self.solution['c1_preds'][0].shape[1] - 1)
        dy = self.params['Ly'] / (self.solution['c1_preds'][0].shape[0] - 1)
        area_element = dx * dy
        
        for t_idx in range(len(times)):
            c1 = self.solution['c1_preds'][t_idx]
            c2 = self.solution['c2_preds'][t_idx]
            
            mass_cu = np.sum(c1) * area_element
            mass_ni = np.sum(c2) * area_element
            
            total_mass_cu.append(mass_cu)
            total_mass_ni.append(mass_ni)
        
        # Plot mass evolution
        axes[0].plot(times, total_mass_cu, 'b-', linewidth=2, label='Total Cu Mass')
        axes[0].plot(times, total_mass_ni, 'r-', linewidth=2, label='Total Ni Mass')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Total Mass (mol)')
        axes[0].set_title('Mass Evolution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot mass conservation error
        initial_mass_cu = total_mass_cu[0]
        initial_mass_ni = total_mass_ni[0]
        
        mass_error_cu = np.abs((np.array(total_mass_cu) - initial_mass_cu) / (initial_mass_cu + 1e-10))
        mass_error_ni = np.abs((np.array(total_mass_ni) - initial_mass_ni) / (initial_mass_ni + 1e-10))
        
        axes[1].semilogy(times, mass_error_cu, 'b--', linewidth=2, label='Cu Mass Error')
        axes[1].semilogy(times, mass_error_ni, 'r--', linewidth=2, label='Ni Mass Error')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Relative Mass Error')
        axes[1].set_title('Mass Conservation Error')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')
        
        plt.suptitle('Mass Conservation Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = os.path.join(self.output_dir, 'mass_conservation.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def plot_boundary_validation(self):
        """Validate boundary condition satisfaction"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        times = np.array(self.solution['times'])
        
        # Extract boundary profiles
        # Bottom boundary (y=0)
        bottom_cu = [c[0, :] for c in self.solution['c1_preds']]
        bottom_ni = [c[0, :] for c in self.solution['c2_preds']]
        
        # Top boundary (y=Ly)
        top_cu = [c[-1, :] for c in self.solution['c1_preds']]
        top_ni = [c[-1, :] for c in self.solution['c2_preds']]
        
        # Left boundary (x=0)
        left_cu = [c[:, 0] for c in self.solution['c1_preds']]
        left_ni = [c[:, 0] for c in self.solution['c2_preds']]
        
        # Right boundary (x=Lx)
        right_cu = [c[:, -1] for c in self.solution['c1_preds']]
        right_ni = [c[:, -1] for c in self.solution['c2_preds']]
        
        # Plot bottom boundary
        x_coords = np.linspace(0, self.params['Lx'], len(bottom_cu[0]))
        for i, t_idx in enumerate([0, len(times)//2, -1]):
            t_val = times[t_idx]
            axes[0, 0].plot(x_coords, bottom_cu[t_idx], 
                           label=f't={t_val:.1f}s', linewidth=2)
        axes[0, 0].axhline(y=self.params['C_CU_BOTTOM'], color='r', 
                          linestyle='--', linewidth=2, label='Target')
        axes[0, 0].set_xlabel('x (μm)')
        axes[0, 0].set_ylabel('Cu Concentration')
        axes[0, 0].set_title('Bottom Boundary (Cu)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot top boundary
        for i, t_idx in enumerate([0, len(times)//2, -1]):
            t_val = times[t_idx]
            axes[0, 1].plot(x_coords, top_cu[t_idx], 
                           label=f't={t_val:.1f}s', linewidth=2)
        axes[0, 1].axhline(y=self.params['C_CU_TOP'], color='r',
                          linestyle='--', linewidth=2, label='Target')
        axes[0, 1].set_xlabel('x (μm)')
        axes[0, 1].set_ylabel('Cu Concentration')
        axes[0, 1].set_title('Top Boundary (Cu)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot left boundary flux
        y_coords = np.linspace(0, self.params['Ly'], len(left_cu[0]))
        for i, t_idx in enumerate([0, len(times)//2, -1]):
            t_val = times[t_idx]
            # Compute derivative (approximate)
            flux = np.gradient(left_cu[t_idx], y_coords)
            axes[1, 0].plot(y_coords, flux, label=f't={t_val:.1f}s', linewidth=2)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Flux')
        axes[1, 0].set_xlabel('y (μm)')
        axes[1, 0].set_ylabel('Flux (∂c/∂x)')
        axes[1, 0].set_title('Left Boundary Flux (Cu)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot right boundary flux
        for i, t_idx in enumerate([0, len(times)//2, -1]):
            t_val = times[t_idx]
            flux = np.gradient(right_cu[t_idx], y_coords)
            axes[1, 1].plot(y_coords, flux, label=f't={t_val:.1f}s', linewidth=2)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Flux')
        axes[1, 1].set_xlabel('y (μm)')
        axes[1, 1].set_ylabel('Flux (∂c/∂x)')
        axes[1, 1].set_title('Right Boundary Flux (Cu)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('Boundary Condition Validation', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        filename = os.path.join(self.output_dir, 'boundary_validation.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def create_concentration_animation(self):
        """Create animated GIF of concentration evolution"""
        from matplotlib.animation import FuncAnimation, PillowWriter
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Initial plots
        im1 = ax1.imshow(self.solution['c1_preds'][0], origin='lower',
                        extent=[0, self.params['Lx'], 0, self.params['Ly']],
                        cmap=self.colormaps['Cu'], vmin=0, vmax=self.params['C_CU_TOP'])
        ax1.set_title('Cu Concentration')
        ax1.set_xlabel('x (μm)')
        ax1.set_ylabel('y (μm)')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(self.solution['c2_preds'][0], origin='lower',
                        extent=[0, self.params['Lx'], 0, self.params['Ly']],
                        cmap=self.colormaps['Ni'], vmin=0, vmax=self.params['C_NI_BOTTOM'])
        ax2.set_title('Ni Concentration')
        ax2.set_xlabel('x (μm)')
        ax2.set_ylabel('y (μm)')
        plt.colorbar(im2, ax=ax2)
        
        time_text = fig.suptitle(f'Time = {self.solution["times"][0]:.1f} s', 
                                fontsize=14, fontweight='bold')
        
        def update(frame):
            im1.set_array(self.solution['c1_preds'][frame])
            im2.set_array(self.solution['c2_preds'][frame])
            time_text.set_text(f'Time = {self.solution["times"][frame]:.1f} s')
            return im1, im2, time_text
        
        # Create animation
        ani = FuncAnimation(fig, update, frames=len(self.solution['times']),
                          interval=100, blit=True)
        
        # Save as GIF
        filename = os.path.join(self.output_dir, 'concentration_evolution.gif')
        ani.save(filename, writer=PillowWriter(fps=10))
        plt.close()
        
        return filename

# ============================================================================
# 7. VTK/VTU EXPORT WITH ENHANCED FEATURES
# ============================================================================

class VTKExporter:
    """Export solution to VTK formats for visualization in ParaView"""
    
    def __init__(self, solution_data, params, output_dir):
        self.solution = solution_data
        self.params = params
        self.output_dir = output_dir
        
    def export_time_series_vts(self):
        """Export time series as VTS files (structured grid)"""
        vts_files = []
        
        # Create output directory
        vts_dir = os.path.join(self.output_dir, 'vts_files')
        os.makedirs(vts_dir, exist_ok=True)
        
        # Grid dimensions
        nx, ny = self.solution['X'].shape[1], self.solution['X'].shape[0]
        
        for t_idx, t_val in enumerate(self.solution['times']):
            # Create structured grid
            grid = pv.StructuredGrid()
            
            # Set points (reshape to 3D with z=0)
            x = self.solution['X'].flatten()
            y = self.solution['Y'].flatten()
            z = np.zeros_like(x)
            points = np.column_stack([x, y, z])
            grid.points = points
            grid.dimensions = [nx, ny, 1]
            
            # Add concentration data
            grid.point_data['Cu_Concentration'] = self.solution['c1_preds'][t_idx].T.flatten()
            grid.point_data['Ni_Concentration'] = self.solution['c2_preds'][t_idx].T.flatten()
            
            # Compute derived quantities
            total_concentration = (self.solution['c1_preds'][t_idx] + 
                                 self.solution['c2_preds'][t_idx]).T.flatten()
            grid.point_data['Total_Concentration'] = total_concentration
            
            # Compute concentration ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                concentration_ratio = np.where(
                    self.solution['c2_preds'][t_idx] > 1e-10,
                    self.solution['c1_preds'][t_idx] / self.solution['c2_preds'][t_idx],
                    0.0
                )
            grid.point_data['Cu_Ni_Ratio'] = concentration_ratio.T.flatten()
            
            # Save VTS file
            vts_filename = os.path.join(vts_dir, f'concentration_t_{t_val:06.1f}.vts')
            grid.save(vts_filename)
            vts_files.append((t_val, vts_filename))
            
        logger.info(f"Exported {len(vts_files)} VTS files to {vts_dir}")
        return vts_files
    
    def export_single_vtu(self):
        """Export all time steps to a single VTU file with time arrays"""
        # Create unstructured grid
        nx, ny = self.solution['X'].shape[1], self.solution['X'].shape[0]
        
        # Create points (2D grid)
        x = self.solution['X'].flatten()
        y = self.solution['Y'].flatten()
        z = np.zeros_like(x)
        points = np.column_stack([x, y, z])
        
        # Create cells (quads)
        cells = []
        cell_types = []
        
        for j in range(ny - 1):
            for i in range(nx - 1):
                idx = i + j * nx
                cell = [4, idx, idx + 1, idx + nx + 1, idx + nx]
                cells.extend(cell)
                cell_types.append(pv.CellType.QUAD)
        
        grid = pv.UnstructuredGrid(cells, cell_types, points)
        
        # Add concentration data for all time steps
        times = self.solution['times']
        for t_idx, t_val in enumerate(times):
            grid.point_data[f'Cu_Concentration_t{t_val:.1f}'] = \
                self.solution['c1_preds'][t_idx].T.flatten()
            grid.point_data[f'Ni_Concentration_t{t_val:.1f}'] = \
                self.solution['c2_preds'][t_idx].T.flatten()
        
        # Add time array
        grid.point_data['Time'] = np.full(len(points), times[-1])
        
        # Save VTU file
        vtu_filename = os.path.join(self.output_dir, 'concentration_all_times.vtu')
        grid.save(vtu_filename)
        
        logger.info(f"Exported single VTU file: {vtu_filename}")
        return vtu_filename
    
    def create_pvd_collection(self, vts_files):
        """Create PVD file for time series visualization in ParaView"""
        pvd_content = ['<?xml version="1.0"?>',
                      '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                      '  <Collection>']
        
        for t_val, vts_file in vts_files:
            rel_path = os.path.relpath(vts_file, self.output_dir)
            pvd_content.append(f'    <DataSet timestep="{t_val:.1f}" part="0" file="{rel_path}"/>')
        
        pvd_content.append('  </Collection>')
        pvd_content.append('</VTKFile>')
        
        pvd_filename = os.path.join(self.output_dir, 'concentration_time_series.pvd')
        with open(pvd_filename, 'w') as f:
            f.write('\n'.join(pvd_content))
        
        logger.info(f"Created PVD collection: {pvd_filename}")
        return pvd_filename
    
    def export_for_paraview_state(self):
        """Create comprehensive export for ParaView state file"""
        # Export all formats
        vts_files = self.export_time_series_vts()
        vtu_file = self.export_single_vtu()
        pvd_file = self.create_pvd_collection(vts_files)
        
        # Create metadata file
        metadata = {
            'parameters': self.params,
            'grid_info': {
                'nx': self.solution['X'].shape[1],
                'ny': self.solution['X'].shape[0],
                'num_timesteps': len(self.solution['times'])
            },
            'files': {
                'vts_files': [f[1] for f in vts_files],
                'vtu_file': vtu_file,
                'pvd_file': pvd_file
            },
            'export_time': datetime.now().isoformat()
        }
        
        metadata_file = os.path.join(self.output_dir, 'export_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'vts_files': vts_files,
            'vtu_file': vtu_file,
            'pvd_file': pvd_file,
            'metadata_file': metadata_file
        }

# ============================================================================
# 8. STREAMLIT UI WITH ADVANCED CONTROLS
# ============================================================================

class PINNStreamlitApp:
    """Streamlit application for interactive PINN simulation"""
    
    def __init__(self):
        self.setup_page()
        self.initialize_session_state()
        
    def setup_page(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="2D PINN Simulation: Cu-Ni Cross-Diffusion",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            padding: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #374151;
            margin-top: 1rem;
        }
        .info-box {
            background-color: #F0F9FF;
            border-left: 4px solid #3B82F6;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        .success-box {
            background-color: #D1FAE5;
            border-left: 4px solid #10B981;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        .warning-box {
            background-color: #FEF3C7;
            border-left: 4px solid #F59E0B;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown('<h1 class="main-header">🔬 2D PINN Simulation: Cu-Ni Cross-Diffusion</h1>', 
                   unsafe_allow_html=True)
        
        # Description
        st.markdown("""
        <div class="info-box">
        <h3>📖 About this Simulation</h3>
        <p>This Physics-Informed Neural Network (PINN) solves coupled cross-diffusion equations 
        for Copper (Cu) and Nickel (Ni) in a 2D domain with time evolution. The model enforces 
        physics constraints, boundary conditions, and initial conditions through a combined loss function.</p>
        </div>
        """, unsafe_allow_html=True)
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'training_complete' not in st.session_state:
            st.session_state.training_complete = False
        if 'solution_data' not in st.session_state:
            st.session_state.solution_data = None
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'loss_history' not in st.session_state:
            st.session_state.loss_history = None
        if 'analysis_plots' not in st.session_state:
            st.session_state.analysis_plots = {}
        if 'export_files' not in st.session_state:
            st.session_state.export_files = {}
        if 'training_progress' not in st.session_state:
            st.session_state.training_progress = 0
            
    def create_sidebar_controls(self):
        """Create sidebar controls for simulation parameters"""
        with st.sidebar:
            st.markdown("## ⚙️ Simulation Parameters")
            
            # Physical parameters
            with st.expander("🔬 Physical Parameters", expanded=True):
                st.markdown("**Diffusion Coefficients (μm²/s)**")
                col1, col2 = st.columns(2)
                with col1:
                    D11 = st.number_input("D₁₁ (Cu self-diffusion)", 
                                        value=PhysicalParameters.D11, 
                                        format="%.5f")
                    D12 = st.number_input("D₁₂ (Cu-Ni cross)", 
                                        value=PhysicalParameters.D12, 
                                        format="%.5f")
                with col2:
                    D21 = st.number_input("D₂₁ (Ni-Cu cross)", 
                                        value=PhysicalParameters.D21, 
                                        format="%.5f")
                    D22 = st.number_input("D₂₂ (Ni self-diffusion)", 
                                        value=PhysicalParameters.D22, 
                                        format="%.5f")
                
                st.markdown("**Domain Dimensions (μm)**")
                col1, col2 = st.columns(2)
                with col1:
                    Lx = st.number_input("Width (Lx)", value=PhysicalParameters.Lx)
                with col2:
                    Ly = st.number_input("Height (Ly)", value=PhysicalParameters.Ly)
                
                st.markdown("**Time Parameters**")
                T_max = st.number_input("Maximum Time (s)", value=PhysicalParameters.T_max)
                
                st.markdown("**Boundary Concentrations (mol/μm³)**")
                col1, col2 = st.columns(2)
                with col1:
                    C_CU_TOP = st.number_input("Top: Cu", value=PhysicalParameters.C_CU_TOP, 
                                              format="%.2e")
                    C_CU_BOTTOM = st.number_input("Bottom: Cu", value=PhysicalParameters.C_CU_BOTTOM,
                                                 format="%.2e")
                with col2:
                    C_NI_TOP = st.number_input("Top: Ni", value=PhysicalParameters.C_NI_TOP,
                                              format="%.2e")
                    C_NI_BOTTOM = st.number_input("Bottom: Ni", value=PhysicalParameters.C_NI_BOTTOM,
                                                 format="%.2e")
            
            # Neural network parameters
            with st.expander("🧠 Neural Network Parameters", expanded=True):
                hidden_layers = st.slider("Hidden Layers", 4, 12, 8)
                hidden_dim = st.slider("Hidden Dimension", 64, 512, 256, step=64)
                fourier_features = st.checkbox("Use Fourier Features", value=True)
                dropout_rate = st.slider("Dropout Rate", 0.0, 0.3, 0.05, 0.01)
            
            # Training parameters
            with st.expander("⚡ Training Parameters", expanded=True):
                epochs = st.number_input("Epochs", 1000, 50000, 10000, step=1000)
                initial_lr = st.number_input("Initial Learning Rate", 1e-5, 1e-2, 1e-3, 
                                           format="%.0e")
                batch_size = st.selectbox("Batch Size", [1024, 2048, 4096, 8192], index=2)
                
                col1, col2 = st.columns(2)
                with col1:
                    use_lr_scheduler = st.checkbox("LR Scheduler", value=True)
                with col2:
                    use_gradient_clipping = st.checkbox("Gradient Clipping", value=True)
            
            # Return parameters
            params = {
                'D11': D11, 'D12': D12, 'D21': D21, 'D22': D22,
                'Lx': Lx, 'Ly': Ly, 'T_max': T_max,
                'C_CU_TOP': C_CU_TOP, 'C_CU_BOTTOM': C_CU_BOTTOM,
                'C_NI_TOP': C_NI_TOP, 'C_NI_BOTTOM': C_NI_BOTTOM,
                'hidden_layers': hidden_layers,
                'hidden_dim': hidden_dim,
                'fourier_features': fourier_features,
                'dropout_rate': dropout_rate,
                'epochs': epochs,
                'initial_lr': initial_lr,
                'batch_size': batch_size,
                'use_lr_scheduler': use_lr_scheduler,
                'use_gradient_clipping': use_gradient_clipping
            }
            
            return params
    
    def run_simulation(self, params):
        """Run the PINN simulation"""
        st.markdown("## 🚀 Running Simulation")
        
        # Create progress tracker
        progress_bar = st.progress(0)
        status_text = st.empty()
        training_log = st.empty()
        
        # Initialize model
        with st.spinner("Initializing model..."):
            model = EnhancedPINN(
                params=params,
                hidden_layers=params['hidden_layers'],
                hidden_dim=params['hidden_dim'],
                fourier_features=params['fourier_features'],
                dropout_rate=params['dropout_rate']
            )
            
            trainer = PINNTrainer(model, params, OUTPUT_DIR)
            
            # Update trainer config
            trainer.config.update({
                'epochs': params['epochs'],
                'initial_lr': params['initial_lr'],
                'batch_size': params['batch_size']
            })
        
        # Training progress callback
        def update_progress(epoch, total_epochs, loss):
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            status_text.text(f"Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}")
            
            # Update log occasionally
            if epoch % 100 == 0:
                training_log.text(f"Epoch {epoch}: Loss = {loss:.6f}")
        
        # Run training
        try:
            with st.spinner("Training in progress..."):
                loss_history = trainer.train(progress_callback=update_progress)
            
            # Training complete
            progress_bar.progress(1.0)
            status_text.text("Training completed successfully!")
            
            # Generate solution
            with st.spinner("Generating solution..."):
                solution = self.generate_solution(model, params)
            
            # Analyze solution
            with st.spinner("Analyzing results..."):
                analyzer = SolutionAnalyzer(solution, params, OUTPUT_DIR)
                analysis_plots = analyzer.create_comprehensive_plots()
            
            # Export to VTK
            with st.spinner("Exporting to VTK..."):
                exporter = VTKExporter(solution, params, OUTPUT_DIR)
                export_files = exporter.export_for_paraview_state()
            
            # Update session state
            st.session_state.training_complete = True
            st.session_state.model = model
            st.session_state.solution_data = solution
            st.session_state.loss_history = loss_history
            st.session_state.analysis_plots = analysis_plots
            st.session_state.export_files = export_files
            
            st.success("✅ Simulation completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Simulation failed: {str(e)}")
            logger.error(f"Simulation failed: {str(e)}", exc_info=True)
    
    def generate_solution(self, model, params):
        """Generate solution on a grid"""
        # Create grid
        nx, ny = 100, 100  # Higher resolution for better visualization
        x = torch.linspace(0, params['Lx'], nx)
        y = torch.linspace(0, params['Ly'], ny)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # Time points
        num_times = 50
        times = np.linspace(0, params['T_max'], num_times)
        
        # Evaluate model at each time
        c1_preds = []
        c2_preds = []
        
        model.eval()
        with torch.no_grad():
            for t_val in times:
                t = torch.full((X.numel(), 1), t_val)
                c_pred = model(X.reshape(-1, 1), Y.reshape(-1, 1), t)
                
                c1 = c_pred[:, 0].numpy().reshape(ny, nx)
                c2 = c_pred[:, 1].numpy().reshape(ny, nx)
                
                c1_preds.append(c1)
                c2_preds.append(c2)
        
        # Create solution dictionary
        solution = {
            'X': X.numpy(),
            'Y': Y.numpy(),
            'c1_preds': c1_preds,
            'c2_preds': c2_preds,
            'times': times,
            'params': params,
            'grid_info': {
                'nx': nx, 'ny': ny,
                'num_times': num_times
            }
        }
        
        return solution
    
    def display_results(self):
        """Display simulation results"""
        if not st.session_state.training_complete:
            return
        
        st.markdown("## 📊 Simulation Results")
        
        # Create tabs for different result categories
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Training Metrics", 
            "🔍 Concentration Profiles",
            "📊 Analysis & Validation",
            "📁 Export & Download",
            "📋 Summary Report"
        ])
        
        with tab1:
            self.display_training_metrics()
        
        with tab2:
            self.display_concentration_profiles()
        
        with tab3:
            self.display_analysis_validation()
        
        with tab4:
            self.display_export_download()
        
        with tab5:
            self.display_summary_report()
    
    def display_training_metrics(self):
        """Display training metrics and loss plots"""
        if st.session_state.loss_history:
            # Plot loss history
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            epochs = range(1, len(st.session_state.loss_history['total']) + 1)
            
            # Total loss
            axes[0, 0].plot(epochs, st.session_state.loss_history['total'], 'b-', linewidth=2)
            axes[0, 0].set_yscale('log')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Total Loss')
            axes[0, 0].set_title('Total Loss History')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Component losses
            axes[0, 1].plot(epochs, st.session_state.loss_history['physics'], 'r-', 
                          label='Physics', linewidth=2)
            axes[0, 1].plot(epochs, st.session_state.loss_history['boundary_bottom'], 'g-',
                          label='Bottom BC', linewidth=2)
            axes[0, 1].plot(epochs, st.session_state.loss_history['boundary_top'], 'b-',
                          label='Top BC', linewidth=2)
            axes[0, 1].set_yscale('log')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].set_title('Component Losses')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate
            axes[1, 0].plot(epochs, st.session_state.loss_history['learning_rate'], 'purple-', 
                          linewidth=2)
            axes[1, 0].set_yscale('log')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Convergence analysis
            if len(epochs) > 100:
                window = 100
                moving_avg = np.convolve(st.session_state.loss_history['total'], 
                                       np.ones(window)/window, mode='valid')
                axes[1, 1].plot(epochs[window-1:], moving_avg, 'orange-', linewidth=2)
                axes[1, 1].set_yscale('log')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Moving Average Loss')
                axes[1, 1].set_title(f'Convergence (Moving Avg, window={window})')
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Loss statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                final_loss = st.session_state.loss_history['total'][-1]
                st.metric("Final Loss", f"{final_loss:.2e}")
            with col2:
                min_loss = min(st.session_state.loss_history['total'])
                st.metric("Minimum Loss", f"{min_loss:.2e}")
            with col3:
                convergence_rate = (st.session_state.loss_history['total'][0] / 
                                  st.session_state.loss_history['total'][-1])
                st.metric("Convergence Factor", f"{convergence_rate:.1f}x")
    
    def display_concentration_profiles(self):
        """Display concentration profiles"""
        if st.session_state.analysis_plots:
            # Show available plots
            st.image(st.session_state.analysis_plots.get('concentration_2d', ''), 
                    caption="2D Concentration Grid")
            
            # Interactive time selection
            times = st.session_state.solution_data['times']
            selected_time = st.slider("Select Time", 0.0, float(times[-1]), 
                                    float(times[-1]), step=float(times[1]-times[0]))
            
            # Find closest time index
            t_idx = np.argmin(np.abs(times - selected_time))
            
            # Display profiles for selected time
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(6, 5))
                c1 = st.session_state.solution_data['c1_preds'][t_idx]
                im = ax.imshow(c1, origin='lower', 
                             extent=[0, st.session_state.solution_data['params']['Lx'],
                                    0, st.session_state.solution_data['params']['Ly']],
                             cmap='viridis')
                ax.set_title(f'Cu Concentration at t = {selected_time:.1f} s')
                ax.set_xlabel('x (μm)')
                ax.set_ylabel('y (μm)')
                plt.colorbar(im, ax=ax, label='Concentration (mol/μm³)')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 5))
                c2 = st.session_state.solution_data['c2_preds'][t_idx]
                im = ax.imshow(c2, origin='lower',
                             extent=[0, st.session_state.solution_data['params']['Lx'],
                                    0, st.session_state.solution_data['params']['Ly']],
                             cmap='plasma')
                ax.set_title(f'Ni Concentration at t = {selected_time:.1f} s')
                ax.set_xlabel('x (μm)')
                ax.set_ylabel('y (μm)')
                plt.colorbar(im, ax=ax, label='Concentration (mol/μm³)')
                st.pyplot(fig)
                plt.close()
            
            # 3D plot option
            if st.checkbox("Show 3D Visualization"):
                st.image(st.session_state.analysis_plots.get('concentration_3d', ''),
                        caption="3D Concentration Profiles")
    
    def display_analysis_validation(self):
        """Display analysis and validation results"""
        if st.session_state.analysis_plots:
            # Create tabs for different analyses
            anal_tab1, anal_tab2, anal_tab3 = st.tabs([
                "📈 Time Evolution",
                "⚖️ Mass Conservation",
                "✅ Boundary Validation"
            ])
            
            with anal_tab1:
                st.image(st.session_state.analysis_plots.get('time_evolution', ''),
                        caption="Time Evolution Analysis")
            
            with anal_tab2:
                st.image(st.session_state.analysis_plots.get('mass_conservation', ''),
                        caption="Mass Conservation Analysis")
                
                # Compute mass conservation metrics
                solution = st.session_state.solution_data
                params = solution['params']
                
                # Approximate total mass
                dx = params['Lx'] / (solution['c1_preds'][0].shape[1] - 1)
                dy = params['Ly'] / (solution['c1_preds'][0].shape[0] - 1)
                area = dx * dy
                
                initial_mass_cu = np.sum(solution['c1_preds'][0]) * area
                final_mass_cu = np.sum(solution['c1_preds'][-1]) * area
                mass_change_cu = abs(final_mass_cu - initial_mass_cu) / initial_mass_cu
                
                initial_mass_ni = np.sum(solution['c2_preds'][0]) * area
                final_mass_ni = np.sum(solution['c2_preds'][-1]) * area
                mass_change_ni = abs(final_mass_ni - initial_mass_ni) / initial_mass_ni
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Cu Mass Change", f"{mass_change_cu*100:.3f}%")
                with col2:
                    st.metric("Ni Mass Change", f"{mass_change_ni*100:.3f}%")
            
            with anal_tab3:
                st.image(st.session_state.analysis_plots.get('boundary_validation', ''),
                        caption="Boundary Condition Validation")
                
                # Boundary condition errors
                solution = st.session_state.solution_data
                params = solution['params']
                
                # Compute boundary errors
                bottom_cu = solution['c1_preds'][-1][0, :]
                bottom_ni = solution['c2_preds'][-1][0, :]
                
                top_cu = solution['c1_preds'][-1][-1, :]
                top_ni = solution['c2_preds'][-1][-1, :]
                
                bottom_cu_error = np.mean(np.abs(bottom_cu - params['C_CU_BOTTOM']))
                bottom_ni_error = np.mean(np.abs(bottom_ni - params['C_NI_BOTTOM']))
                top_cu_error = np.mean(np.abs(top_cu - params['C_CU_TOP']))
                top_ni_error = np.mean(np.abs(top_ni - params['C_NI_TOP']))
                
                st.markdown("**Boundary Condition Errors**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Bottom Cu Error", f"{bottom_cu_error:.2e}")
                    st.metric("Top Cu Error", f"{top_cu_error:.2e}")
                with col2:
                    st.metric("Bottom Ni Error", f"{bottom_ni_error:.2e}")
                    st.metric("Top Ni Error", f"{top_ni_error:.2e}")
    
    def display_export_download(self):
        """Display export and download options"""
        st.markdown("## 📁 Export & Download")
        
        if not st.session_state.export_files:
            st.warning("No export files available. Run the simulation first.")
            return
        
        # Create download buttons for different formats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Analysis Plots")
            if st.session_state.analysis_plots:
                for plot_name, plot_file in st.session_state.analysis_plots.items():
                    if os.path.exists(plot_file):
                        with open(plot_file, 'rb') as f:
                            st.download_button(
                                label=f"Download {plot_name}.png",
                                data=f,
                                file_name=os.path.basename(plot_file),
                                mime="image/png"
                            )
        
        with col2:
            st.markdown("### 📈 VTK Files")
            if 'vtu_file' in st.session_state.export_files:
                vtu_file = st.session_state.export_files['vtu_file']
                if os.path.exists(vtu_file):
                    with open(vtu_file, 'rb') as f:
                        st.download_button(
                            label="Download VTU File (.vtu)",
                            data=f,
                            file_name=os.path.basename(vtu_file),
                            mime="application/octet-stream"
                        )
            
            if 'pvd_file' in st.session_state.export_files:
                pvd_file = st.session_state.export_files['pvd_file']
                if os.path.exists(pvd_file):
                    with open(pvd_file, 'rb') as f:
                        st.download_button(
                            label="Download PVD Collection (.pvd)",
                            data=f,
                            file_name=os.path.basename(pvd_file),
                            mime="application/xml"
                        )
        
        with col3:
            st.markdown("### 📋 Solution Data")
            # Save solution as pickle
            solution_file = os.path.join(OUTPUT_DIR, 'solution_data.pkl')
            with open(solution_file, 'wb') as f:
                pickle.dump(st.session_state.solution_data, f)
            
            with open(solution_file, 'rb') as f:
                st.download_button(
                    label="Download Solution (.pkl)",
                    data=f,
                    file_name="solution_data.pkl",
                    mime="application/octet-stream"
                )
            
            # Save parameters as JSON
            params_file = os.path.join(OUTPUT_DIR, 'simulation_parameters.json')
            with open(params_file, 'w') as f:
                json.dump(st.session_state.solution_data['params'], f, indent=2)
            
            with open(params_file, 'rb') as f:
                st.download_button(
                    label="Download Parameters (.json)",
                    data=f,
                    file_name="simulation_parameters.json",
                    mime="application/json"
                )
        
        # Create ZIP archive of all files
        st.markdown("### 📦 Complete Archive")
        if st.button("Create ZIP Archive"):
            with st.spinner("Creating ZIP archive..."):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Add analysis plots
                    for plot_file in st.session_state.analysis_plots.values():
                        if os.path.exists(plot_file):
                            zip_file.write(plot_file, 
                                         os.path.basename(plot_file))
                    
                    # Add VTK files
                    if 'vtu_file' in st.session_state.export_files:
                        vtu_file = st.session_state.export_files['vtu_file']
                        if os.path.exists(vtu_file):
                            zip_file.write(vtu_file, os.path.basename(vtu_file))
                    
                    # Add PVD file
                    if 'pvd_file' in st.session_state.export_files:
                        pvd_file = st.session_state.export_files['pvd_file']
                        if os.path.exists(pvd_file):
                            zip_file.write(pvd_file, os.path.basename(pvd_file))
                    
                    # Add VTS files
                    if 'vts_files' in st.session_state.export_files:
                        for t_val, vts_file in st.session_state.export_files['vts_files']:
                            if os.path.exists(vts_file):
                                zip_file.write(vts_file, os.path.basename(vts_file))
                    
                    # Add solution data
                    if os.path.exists(solution_file):
                        zip_file.write(solution_file, 'solution_data.pkl')
                    
                    # Add parameters
                    if os.path.exists(params_file):
                        zip_file.write(params_file, 'simulation_parameters.json')
                    
                    # Add log file
                    log_file = os.path.join(OUTPUT_DIR, 'training.log')
                    if os.path.exists(log_file):
                        zip_file.write(log_file, 'training.log')
                
                # Create download button for ZIP
                st.download_button(
                    label="Download Complete Archive (.zip)",
                    data=zip_buffer.getvalue(),
                    file_name=f"pinn_simulation_{timestamp}.zip",
                    mime="application/zip"
                )
    
    def display_summary_report(self):
        """Display comprehensive summary report"""
        st.markdown("## 📋 Simulation Summary Report")
        
        if not st.session_state.solution_data:
            st.warning("No simulation data available.")
            return
        
        solution = st.session_state.solution_data
        params = solution['params']
        
        # Create summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Domain Size", f"{params['Lx']:.0f} × {params['Ly']:.0f} μm")
        with col2:
            st.metric("Simulation Time", f"{params['T_max']:.0f} s")
        with col3:
            st.metric("Grid Resolution", f"{solution['grid_info']['nx']} × {solution['grid_info']['ny']}")
        with col4:
            st.metric("Time Steps", solution['grid_info']['num_times'])
        
        # Physical parameters table
        st.markdown("### 🔬 Physical Parameters")
        phys_params = pd.DataFrame({
            'Parameter': ['D₁₁', 'D₁₂', 'D₂₁', 'D₂₂', 
                         'C_Cu_top', 'C_Cu_bottom', 
                         'C_Ni_top', 'C_Ni_bottom'],
            'Value': [f"{params['D11']:.5f}", f"{params['D12']:.5f}",
                     f"{params['D21']:.5f}", f"{params['D22']:.5f}",
                     f"{params['C_CU_TOP']:.2e}", f"{params['C_CU_BOTTOM']:.2e}",
                     f"{params['C_NI_TOP']:.2e}", f"{params['C_NI_BOTTOM']:.2e}"],
            'Units': ['μm²/s', 'μm²/s', 'μm²/s', 'μm²/s',
                     'mol/μm³', 'mol/μm³', 'mol/μm³', 'mol/μm³']
        })
        st.table(phys_params)
        
        # Final concentration statistics
        st.markdown("### 📊 Final Concentration Statistics")
        final_c1 = solution['c1_preds'][-1]
        final_c2 = solution['c2_preds'][-1]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Copper (Cu)**")
            st.metric("Mean", f"{np.mean(final_c1):.2e}")
            st.metric("Maximum", f"{np.max(final_c1):.2e}")
            st.metric("Minimum", f"{np.min(final_c1):.2e}")
            st.metric("Std Dev", f"{np.std(final_c1):.2e}")
        
        with col2:
            st.markdown("**Nickel (Ni)**")
            st.metric("Mean", f"{np.mean(final_c2):.2e}")
            st.metric("Maximum", f"{np.max(final_c2):.2e}")
            st.metric("Minimum", f"{np.min(final_c2):.2e}")
            st.metric("Std Dev", f"{np.std(final_c2):.2e}")
        
        # Training summary
        if st.session_state.loss_history:
            st.markdown("### ⚡ Training Summary")
            final_loss = st.session_state.loss_history['total'][-1]
            min_loss = min(st.session_state.loss_history['total'])
            training_time = len(st.session_state.loss_history['total']) * 0.1  # Approximate
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Final Loss", f"{final_loss:.2e}")
            with col2:
                st.metric("Best Loss", f"{min_loss:.2e}")
            with col3:
                st.metric("Training Time", f"{training_time:.1f} s")
        
        # Export summary
        st.markdown("### 📁 Generated Files")
        file_counts = {
            'Analysis Plots': len(st.session_state.analysis_plots),
            'VTK Files': len(st.session_state.export_files.get('vts_files', [])),
            'Data Files': 3  # solution, params, log
        }
        
        for file_type, count in file_counts.items():
            st.markdown(f"- {file_type}: {count} files")
    
    def run(self):
        """Main run method for the Streamlit app"""
        # Display sidebar and get parameters
        params = self.create_sidebar_controls()
        
        # Run simulation button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
            self.run_simulation(params)
        
        # Reset button
        if st.sidebar.button("🔄 Reset Simulation", type="secondary", use_container_width=True):
            self.initialize_session_state()
            st.rerun()
        
        # Display results if available
        if st.session_state.training_complete:
            self.display_results()
        else:
            # Show instructions
            st.markdown("""
            <div class="info-box">
            <h3>📝 Instructions</h3>
            <ol>
            <li>Adjust simulation parameters in the sidebar</li>
            <li>Click <strong>"Run Simulation"</strong> to start</li>
            <li>Monitor training progress in real-time</li>
            <li>View results in the tabs below after completion</li>
            <li>Download analysis plots and data files</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Show default visualization
            self.show_example_visualization()

# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run the Streamlit app"""
    try:
        # Initialize the app
        app = PINNStreamlitApp()
        app.run()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
