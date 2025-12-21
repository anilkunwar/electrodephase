import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import io
import os
from scipy.special import erf
import time
from datetime import datetime
import json

# =====================================================
# PHYSICAL CONSTANTS FOR LiFePO₄
# =====================================================

class PhysicalConstants:
    """Physical constants for LiFePO₄ system"""
    def __init__(self):
        # Fundamental constants
        self.R = 8.314462618  # J/(mol·K)
        self.F = 96485.33212  # C/mol
        self.k_B = 1.380649e-23  # J/K
        
        # LiFePO₄ specific parameters
        self.T = 298.15  # K
        self.V_m = 3.0e-5  # m³/mol (molar volume)
        
        # Phase compositions
        self.c_alpha = 0.03  # FePO₄-rich phase (x in LiₓFePO₄)
        self.c_beta = 0.97   # LiFePO₄-rich phase
        self.c_avg = 0.5     # Average composition
        
        # Material properties
        self.Ω = 55e3  # J/mol (regular solution parameter)
        self.W = self.Ω / self.V_m  # J/m³ (double-well height)
        self.kappa = 2.0 * self.W * (1e-9)**2  # J/m (gradient coefficient, ~1 nm interface)
        self.M = 1e-14 / (self.W * (1e-9)**2)  # m⁵/(J·s) (mobility)
        
        # Reference voltage
        self.V0 = 3.42  # V vs Li/Li⁺
        
        # Electrochemical kinetics
        self.i0 = 1e-3  # A/m² (exchange current density)
        self.alpha = 0.5  # Charge transfer coefficient
        
        # Particle properties (for spherical coordinates)
        self.particle_radius = 50e-9  # 50 nm
        
        # Domain dimensions
        self.Lx = 100e-9  # 100 nm
        self.Ly = 100e-9  # 100 nm
        self.T_max = 3600.0  # 1 hour simulation

# =====================================================
# PINN MODEL ARCHITECTURE
# =====================================================

class PhaseFieldPINN(nn.Module):
    """Physics-Informed Neural Network for LiFePO₄ phase decomposition"""
    
    def __init__(self, constants, geometry='cartesian_2d', include_voltage=True):
        """
        Args:
            constants: PhysicalConstants object
            geometry: 'cartesian_2d', 'cartesian_1d', or 'spherical_1d'
            include_voltage: Whether to include voltage prediction
        """
        super().__init__()
        
        self.constants = constants
        self.geometry = geometry
        self.include_voltage = include_voltage
        
        # Input dimension depends on geometry
        if geometry == 'cartesian_2d':
            input_dim = 3  # (x, y, t)
        elif geometry == 'cartesian_1d':
            input_dim = 2  # (x, t)
        elif geometry == 'spherical_1d':
            input_dim = 2  # (r, t)
        else:
            raise ValueError(f"Unknown geometry: {geometry}")
        
        # Network architecture
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        
        # Output networks
        self.concentration_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.chemical_potential_head = nn.Sequential(
            nn.Linear(128, 1)
        )
        
        if self.include_voltage:
            self.voltage_head = nn.Sequential(
                nn.Linear(128, 1)
            )
        
        # Scale outputs to physical ranges
        self.c_scale = constants.c_beta - constants.c_alpha
        self.c_offset = constants.c_alpha
    
    def forward(self, *inputs):
        """Forward pass of the network"""
        # Combine inputs based on geometry
        if self.geometry == 'cartesian_2d':
            x, y, t = inputs
            inputs_tensor = torch.cat([x, y, t], dim=1)
        else:
            x, t = inputs
            inputs_tensor = torch.cat([x, t], dim=1)
        
        # Shared features
        features = self.shared_net(inputs_tensor)
        
        # Concentration output (scaled to [c_alpha, c_beta])
        c_raw = self.concentration_head(features)
        c = self.c_offset + self.c_scale * c_raw
        
        # Chemical potential output
        mu = self.chemical_potential_head(features)
        
        outputs = {'c': c, 'mu': mu}
        
        # Voltage output if enabled
        if self.include_voltage:
            V = self.voltage_head(features)
            outputs['V'] = V
        
        return outputs

# =====================================================
# PHYSICS LOSS FUNCTIONS
# =====================================================

def compute_double_well_derivative(c, W):
    """Compute derivative of double-well free energy: df/dc = 2Wc(1-c)(1-2c)"""
    return 2 * W * c * (1 - c) * (1 - 2 * c)

def compute_laplacian_2d(f, x, y):
    """Compute 2D Laplacian using automatic differentiation"""
    # First derivatives
    f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0]
    f_y = torch.autograd.grad(f, y, grad_outputs=torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    f_xx = torch.autograd.grad(f_x, x, grad_outputs=torch.ones_like(f_x),
                              create_graph=True, retain_graph=True)[0]
    f_yy = torch.autograd.grad(f_y, y, grad_outputs=torch.ones_like(f_y),
                              create_graph=True, retain_graph=True)[0]
    
    return f_xx + f_yy

def compute_laplacian_1d(f, x):
    """Compute 1D Laplacian"""
    f_x = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0]
    f_xx = torch.autograd.grad(f_x, x, grad_outputs=torch.ones_like(f_x),
                              create_graph=True, retain_graph=True)[0]
    return f_xx

def compute_spherical_laplacian(f, r):
    """Compute Laplacian in spherical coordinates (1/r² ∂/∂r (r² ∂f/∂r))"""
    f_r = torch.autograd.grad(f, r, grad_outputs=torch.ones_like(f),
                             create_graph=True, retain_graph=True)[0]
    
    # Compute r² * f_r
    r2_f_r = r**2 * f_r
    
    # Derivative of r² * f_r with respect to r
    r2_f_r_r = torch.autograd.grad(r2_f_r, r, grad_outputs=torch.ones_like(r2_f_r),
                                  create_graph=True, retain_graph=True)[0]
    
    # Spherical Laplacian: (1/r²) * ∂/∂r (r² ∂f/∂r)
    lap_f = r2_f_r_r / (r**2 + 1e-12)  # Add small epsilon to avoid division by zero
    
    return lap_f

def pde_loss_cartesian_2d(model, x, y, t, constants):
    """Compute Cahn-Hilliard PDE loss for 2D Cartesian coordinates"""
    # Get predictions
    outputs = model(x, y, t)
    c = outputs['c']
    mu_pred = outputs['mu']
    
    # Compute chemical potential from physics
    df_dc = compute_double_well_derivative(c, constants.W)
    lap_c = compute_laplacian_2d(c, x, y)
    mu_physics = df_dc - constants.kappa * lap_c
    
    # Chemical potential residual
    mu_residual = mu_pred - mu_physics
    
    # Time derivative of concentration
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                             create_graph=True, retain_graph=True)[0]
    
    # Spatial derivatives of chemical potential
    mu_x = torch.autograd.grad(mu_pred, x, grad_outputs=torch.ones_like(mu_pred),
                              create_graph=True, retain_graph=True)[0]
    mu_y = torch.autograd.grad(mu_pred, y, grad_outputs=torch.ones_like(mu_pred),
                              create_graph=True, retain_graph=True)[0]
    
    mu_xx = torch.autograd.grad(mu_x, x, grad_outputs=torch.ones_like(mu_x),
                               create_graph=True, retain_graph=True)[0]
    mu_yy = torch.autograd.grad(mu_y, y, grad_outputs=torch.ones_like(mu_y),
                               create_graph=True, retain_graph=True)[0]
    
    # Cahn-Hilliard evolution: ∂c/∂t = M∇²μ
    evolution_residual = c_t - constants.M * (mu_xx + mu_yy)
    
    # Total PDE loss
    pde_loss = torch.mean(mu_residual**2) + torch.mean(evolution_residual**2)
    
    return pde_loss

def pde_loss_spherical_1d(model, r, t, constants):
    """Compute Cahn-Hilliard PDE loss for 1D spherical coordinates"""
    outputs = model(r, t)
    c = outputs['c']
    mu_pred = outputs['mu']
    
    # Chemical potential from physics
    df_dc = compute_double_well_derivative(c, constants.W)
    lap_c = compute_spherical_laplacian(c, r)
    mu_physics = df_dc - constants.kappa * lap_c
    
    # Chemical potential residual
    mu_residual = mu_pred - mu_physics
    
    # Time derivative
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c),
                             create_graph=True, retain_graph=True)[0]
    
    # Spherical Laplacian of chemical potential
    lap_mu = compute_spherical_laplacian(mu_pred, r)
    
    # Evolution equation in spherical coordinates
    evolution_residual = c_t - constants.M * lap_mu
    
    # Total PDE loss
    pde_loss = torch.mean(mu_residual**2) + torch.mean(evolution_residual**2)
    
    return pde_loss

def boundary_conditions_loss(model, constants, geometry='cartesian_2d'):
    """Compute boundary condition losses"""
    losses = []
    
    if geometry == 'cartesian_2d':
        # No-flux boundary conditions: ∇c·n = 0, ∇μ·n = 0
        
        # Left boundary (x = 0)
        x_left = torch.zeros(100, 1, requires_grad=True)
        y_left = torch.rand(100, 1, requires_grad=True) * constants.Ly
        t_left = torch.rand(100, 1, requires_grad=True) * constants.T_max
        
        outputs_left = model(x_left, y_left, t_left)
        c_left = outputs_left['c']
        mu_left = outputs_left['mu']
        
        c_x_left = torch.autograd.grad(c_left, x_left, grad_outputs=torch.ones_like(c_left),
                                      create_graph=True, retain_graph=True)[0]
        mu_x_left = torch.autograd.grad(mu_left, x_left, grad_outputs=torch.ones_like(mu_left),
                                       create_graph=True, retain_graph=True)[0]
        
        losses.append(torch.mean(c_x_left**2 + mu_x_left**2))
        
        # Right boundary (x = Lx)
        x_right = torch.full((100, 1), constants.Lx, requires_grad=True)
        y_right = torch.rand(100, 1, requires_grad=True) * constants.Ly
        t_right = torch.rand(100, 1, requires_grad=True) * constants.T_max
        
        outputs_right = model(x_right, y_right, t_right)
        c_right = outputs_right['c']
        mu_right = outputs_right['mu']
        
        c_x_right = torch.autograd.grad(c_right, x_right, grad_outputs=torch.ones_like(c_right),
                                       create_graph=True, retain_graph=True)[0]
        mu_x_right = torch.autograd.grad(mu_right, x_right, grad_outputs=torch.ones_like(mu_right),
                                        create_graph=True, retain_graph=True)[0]
        
        losses.append(torch.mean(c_x_right**2 + mu_x_right**2))
        
        # Top boundary (y = Ly)
        x_top = torch.rand(100, 1, requires_grad=True) * constants.Lx
        y_top = torch.full((100, 1), constants.Ly, requires_grad=True)
        t_top = torch.rand(100, 1, requires_grad=True) * constants.T_max
        
        outputs_top = model(x_top, y_top, t_top)
        c_top = outputs_top['c']
        mu_top = outputs_top['mu']
        
        c_y_top = torch.autograd.grad(c_top, y_top, grad_outputs=torch.ones_like(c_top),
                                     create_graph=True, retain_graph=True)[0]
        mu_y_top = torch.autograd.grad(mu_top, y_top, grad_outputs=torch.ones_like(mu_top),
                                      create_graph=True, retain_graph=True)[0]
        
        losses.append(torch.mean(c_y_top**2 + mu_y_top**2))
        
        # Bottom boundary (y = 0)
        x_bottom = torch.rand(100, 1, requires_grad=True) * constants.Lx
        y_bottom = torch.zeros(100, 1, requires_grad=True)
        t_bottom = torch.rand(100, 1, requires_grad=True) * constants.T_max
        
        outputs_bottom = model(x_bottom, y_bottom, t_bottom)
        c_bottom = outputs['c']
        mu_bottom = outputs['mu']
        
        c_y_bottom = torch.autograd.grad(c_bottom, y_bottom, grad_outputs=torch.ones_like(c_bottom),
                                        create_graph=True, retain_graph=True)[0]
        mu_y_bottom = torch.autograd.grad(mu_bottom, y_bottom, grad_outputs=torch.ones_like(mu_bottom),
                                         create_graph=True, retain_graph=True)[0]
        
        losses.append(torch.mean(c_y_bottom**2 + mu_y_bottom**2))
        
    elif geometry == 'spherical_1d':
        # Center boundary (r = 0): symmetry condition
        r_center = torch.zeros(100, 1, requires_grad=True)
        t_center = torch.rand(100, 1, requires_grad=True) * constants.T_max
        
        outputs_center = model(r_center, t_center)
        c_center = outputs_center['c']
        
        # Use finite difference to compute gradient at center
        r_small = torch.full((100, 1), 1e-9, requires_grad=True)
        outputs_small = model(r_small, t_center)
        c_small = outputs_small['c']
        
        # Approximate gradient at center using forward difference
        c_grad_center = (c_small - c_center) / 1e-9
        losses.append(torch.mean(c_grad_center**2))
        
        # Surface boundary (r = R): Butler-Volmer kinetics
        r_surface = torch.full((100, 1), constants.particle_radius, requires_grad=True)
        t_surface = torch.rand(100, 1, requires_grad=True) * constants.T_max
        
        outputs_surface = model(r_surface, t_surface)
        c_surface = outputs_surface['c']
        mu_surface = outputs_surface['mu']
        
        # Compute flux from Butler-Volmer equation
        # j = i0 * [exp(αFη/RT) - exp(-(1-α)Fη/RT)]
        # where η = V - V_eq, V_eq = V0 - (1/F) * μ_surface
        if model.include_voltage:
            V_pred = outputs_surface['V']
            V_eq = constants.V0 - (1/constants.F) * mu_surface
            eta = V_pred - V_eq
            
            # Butler-Volmer flux (converted to mol/m²·s)
            F_by_RT = constants.F / (constants.R * constants.T)
            j_BV = constants.i0 / constants.F * (
                torch.exp(constants.alpha * F_by_RT * eta) -
                torch.exp(-(1 - constants.alpha) * F_by_RT * eta)
            )
            
            # Compute actual flux from model (M * ∂μ/∂r at surface)
            mu_r_surface = torch.autograd.grad(mu_surface, r_surface,
                                              grad_outputs=torch.ones_like(mu_surface),
                                              create_graph=True, retain_graph=True)[0]
            actual_flux = constants.M * mu_r_surface
            
            # Flux matching loss
            flux_loss = torch.mean((actual_flux - j_BV)**2)
            losses.append(flux_loss)
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)

def initial_condition_loss(model, constants, geometry='cartesian_2d'):
    """Compute initial condition loss"""
    if geometry == 'cartesian_2d':
        x = torch.rand(500, 1, requires_grad=True) * constants.Lx
        y = torch.rand(500, 1, requires_grad=True) * constants.Ly
        t = torch.zeros(500, 1, requires_grad=True)
        
        outputs = model(x, y, t)
        c_pred = outputs['c']
        
        # Initial condition: uniform concentration with small fluctuations
        c0 = constants.c_avg + 0.05 * torch.randn_like(c_pred)
        
        return torch.mean((c_pred - c0)**2)
    
    elif geometry == 'spherical_1d':
        r = torch.rand(500, 1, requires_grad=True) * constants.particle_radius
        t = torch.zeros(500, 1, requires_grad=True)
        
        outputs = model(r, t)
        c_pred = outputs['c']
        
        # Initial condition: linear gradient from center to surface
        r_norm = r / constants.particle_radius
        c0 = constants.c_alpha + (constants.c_beta - constants.c_alpha) * r_norm
        
        return torch.mean((c_pred - c0)**2)
    
    return torch.tensor(0.0)

def voltage_constraint_loss(model, constants, geometry='cartesian_2d'):
    """Compute voltage constraint loss: V = V0 - (1/F) * ⟨μ⟩"""
    if not model.include_voltage:
        return torch.tensor(0.0)
    
    if geometry == 'cartesian_2d':
        x = torch.rand(200, 1, requires_grad=True) * constants.Lx
        y = torch.rand(200, 1, requires_grad=True) * constants.Ly
        t = torch.rand(200, 1, requires_grad=True) * constants.T_max
        
        outputs = model(x, y, t)
        V_pred = outputs['V']
        mu_pred = outputs['mu']
        
        # Spatial average of chemical potential
        mu_avg = torch.mean(mu_pred)
        
        # Voltage constraint
        V_constraint = constants.V0 - (1/constants.F) * mu_avg
        
        return torch.mean((V_pred - V_constraint)**2)
    
    return torch.tensor(0.0)

def experimental_data_loss(model, experimental_data, constants):
    """Compute loss from experimental data if available"""
    if experimental_data is None:
        return torch.tensor(0.0)
    
    losses = []
    
    for data_point in experimental_data:
        if 'type' not in data_point:
            continue
            
        if data_point['type'] == 'voltage':
            # Voltage vs time data
            t_data = torch.tensor(data_point['time'], dtype=torch.float32).reshape(-1, 1)
            
            if model.geometry == 'cartesian_2d':
                # For voltage, we can use arbitrary spatial points
                x = torch.ones_like(t_data) * constants.Lx / 2
                y = torch.ones_like(t_data) * constants.Ly / 2
                outputs = model(x, y, t_data)
            else:
                r = torch.ones_like(t_data) * constants.particle_radius / 2
                outputs = model(r, t_data)
            
            V_pred = outputs['V']
            V_data = torch.tensor(data_point['voltage'], dtype=torch.float32).reshape(-1, 1)
            
            losses.append(torch.mean((V_pred - V_data)**2))
            
        elif data_point['type'] == 'concentration_profile':
            # Concentration profile at specific time
            t_data = torch.full((len(data_point['x']), 1), data_point['time'], dtype=torch.float32)
            
            if model.geometry == 'cartesian_2d':
                x = torch.tensor(data_point['x'], dtype=torch.float32).reshape(-1, 1)
                y = torch.tensor(data_point['y'], dtype=torch.float32).reshape(-1, 1)
                outputs = model(x, y, t_data)
            else:
                r = torch.tensor(data_point['r'], dtype=torch.float32).reshape(-1, 1)
                outputs = model(r, t_data)
            
            c_pred = outputs['c']
            c_data = torch.tensor(data_point['concentration'], dtype=torch.float32).reshape(-1, 1)
            
            losses.append(torch.mean((c_pred - c_data)**2))
    
    return torch.stack(losses).mean() if losses else torch.tensor(0.0)

# =====================================================
# TRAINING FUNCTION
# =====================================================

def train_pinn_model(constants, geometry='cartesian_2d', include_voltage=True,
                    experimental_data=None, epochs=5000, lr=1e-3):
    """Train the PINN model"""
    
    # Initialize model
    model = PhaseFieldPINN(constants, geometry, include_voltage)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    
    # Loss history
    history = {
        'total_loss': [],
        'pde_loss': [],
        'bc_loss': [],
        'ic_loss': [],
        'voltage_loss': [],
        'data_loss': []
    }
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Sample collocation points
        if geometry == 'cartesian_2d':
            # PDE points
            x_pde = torch.rand(1000, 1, requires_grad=True) * constants.Lx
            y_pde = torch.rand(1000, 1, requires_grad=True) * constants.Ly
            t_pde = torch.rand(1000, 1, requires_grad=True) * constants.T_max
            
            # Compute losses
            pde_loss = pde_loss_cartesian_2d(model, x_pde, y_pde, t_pde, constants)
            
        elif geometry == 'spherical_1d':
            # PDE points
            r_pde = torch.rand(1000, 1, requires_grad=True) * constants.particle_radius
            t_pde = torch.rand(1000, 1, requires_grad=True) * constants.T_max
            
            # Compute losses
            pde_loss = pde_loss_spherical_1d(model, r_pde, t_pde, constants)
        
        # Boundary conditions loss
        bc_loss = boundary_conditions_loss(model, constants, geometry)
        
        # Initial condition loss
        ic_loss = initial_condition_loss(model, constants, geometry)
        
        # Voltage constraint loss
        voltage_loss = voltage_constraint_loss(model, constants, geometry)
        
        # Experimental data loss
        data_loss = experimental_data_loss(model, experimental_data, constants)
        
        # Total loss with weights
        total_loss = (1.0 * pde_loss + 
                     10.0 * bc_loss + 
                     10.0 * ic_loss + 
                     5.0 * voltage_loss + 
                     2.0 * data_loss)
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(total_loss)
        
        # Record history
        history['total_loss'].append(total_loss.item())
        history['pde_loss'].append(pde_loss.item())
        history['bc_loss'].append(bc_loss.item())
        history['ic_loss'].append(ic_loss.item())
        history['voltage_loss'].append(voltage_loss.item())
        history['data_loss'].append(data_loss.item())
        
        # Print progress
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6f}, "
                  f"PDE Loss = {pde_loss.item():.6f}")
    
    return model, history

# =====================================================
# VISUALIZATION FUNCTIONS
# =====================================================

def plot_loss_history(history):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    loss_types = ['total_loss', 'pde_loss', 'bc_loss', 'ic_loss', 'voltage_loss', 'data_loss']
    titles = ['Total Loss', 'PDE Loss', 'BC Loss', 'IC Loss', 'Voltage Loss', 'Data Loss']
    
    for idx, (loss_type, title) in enumerate(zip(loss_types, titles)):
        if loss_type in history and history[loss_type]:
            axes[idx].semilogy(history[loss_type])
            axes[idx].set_title(title)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_concentration_profile_2d(model, constants, t_value):
    """Plot 2D concentration profile at specific time"""
    # Create grid
    x = torch.linspace(0, constants.Lx, 100)
    y = torch.linspace(0, constants.Ly, 100)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Flatten for model input
    X_flat = X.reshape(-1, 1)
    Y_flat = Y.reshape(-1, 1)
    t = torch.full_like(X_flat, t_value)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(X_flat, Y_flat, t)
        c_pred = outputs['c'].reshape(100, 100).numpy()
    
    # Create custom colormap for LiFePO₄
    colors = [(0.8, 0.1, 0.1), (0.9, 0.9, 0.1), (0.1, 0.5, 0.8)]  # Red -> Yellow -> Blue
    cmap = LinearSegmentedColormap.from_list('lifepo4', colors, N=256)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 2D contour plot
    im1 = ax1.imshow(c_pred.T, extent=[0, constants.Lx*1e9, 0, constants.Ly*1e9],
                     origin='lower', cmap=cmap, aspect='auto')
    ax1.set_xlabel('x (nm)')
    ax1.set_ylabel('y (nm)')
    ax1.set_title(f'Li Concentration in LiₓFePO₄ at t = {t_value:.1f} s')
    plt.colorbar(im1, ax=ax1, label='x in LiₓFePO₄')
    
    # Add phase boundary indicators
    ax1.contour(c_pred.T, levels=[constants.c_alpha, constants.c_beta], 
                colors=['white', 'black'], linewidths=1, alpha=0.5)
    
    # Line profile at y = Ly/2
    y_mid_idx = 50
    ax2.plot(x.numpy()*1e9, c_pred[:, y_mid_idx], 'b-', linewidth=2)
    ax2.axhline(constants.c_alpha, color='r', linestyle='--', alpha=0.7, label='FePO₄')
    ax2.axhline(constants.c_beta, color='g', linestyle='--', alpha=0.7, label='LiFePO₄')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('Li Concentration (x)')
    ax2.set_title('Line Profile at y = L/2')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_spherical_profile(model, constants, t_values):
    """Plot concentration profiles in spherical coordinates"""
    fig, axes = plt.subplots(1, len(t_values), figsize=(5*len(t_values), 4))
    
    if len(t_values) == 1:
        axes = [axes]
    
    for idx, t_val in enumerate(t_values):
        # Create radial grid
        r = torch.linspace(0, constants.particle_radius, 100).reshape(-1, 1)
        t = torch.full_like(r, t_val)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(r, t)
            c_pred = outputs['c'].numpy().flatten()
            
            if model.include_voltage:
                V_pred = outputs['V'].numpy().flatten()[0]
        
        # Plot
        ax = axes[idx]
        ax.plot(r.numpy().flatten()*1e9, c_pred, 'b-', linewidth=2)
        ax.axhline(constants.c_alpha, color='r', linestyle='--', alpha=0.7, label='FePO₄')
        ax.axhline(constants.c_beta, color='g', linestyle='--', alpha=0.7, label='LiFePO₄')
        
        ax.set_xlabel('Radius (nm)')
        ax.set_ylabel('Li Concentration (x)')
        ax.set_title(f't = {t_val:.1f} s' + 
                    (f', V = {V_pred:.3f} V' if model.include_voltage else ''))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_voltage_profile(model, constants, geometry='cartesian_2d'):
    """Plot voltage vs time profile"""
    # Create time array
    t_values = torch.linspace(0, constants.T_max, 100).reshape(-1, 1)
    
    if geometry == 'cartesian_2d':
        # Use center point for voltage measurement
        x = torch.ones_like(t_values) * constants.Lx / 2
        y = torch.ones_like(t_values) * constants.Ly / 2
        
        with torch.no_grad():
            outputs = model(x, y, t_values)
            V_pred = outputs['V'].numpy().flatten()
    else:
        # Use surface point for spherical
        r = torch.ones_like(t_values) * constants.particle_radius
        
        with torch.no_grad():
            outputs = model(r, t_values)
            V_pred = outputs['V'].numpy().flatten()
    
    # Also compute equilibrium voltage from average chemical potential
    V_eq = []
    for t_val in t_values:
        # Sample multiple points for average
        if geometry == 'cartesian_2d':
            x_sample = torch.rand(100, 1) * constants.Lx
            y_sample = torch.rand(100, 1) * constants.Ly
            t_sample = torch.full_like(x_sample, t_val.item())
            outputs_sample = model(x_sample, y_sample, t_sample)
        else:
            r_sample = torch.rand(100, 1) * constants.particle_radius
            t_sample = torch.full_like(r_sample, t_val.item())
            outputs_sample = model(r_sample, t_sample)
        
        mu_avg = outputs_sample['mu'].mean().item()
        V_eq.append(constants.V0 - (1/constants.F) * mu_avg)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_values.numpy().flatten(), V_pred, 'b-', linewidth=2, label='PINN Prediction')
    ax.plot(t_values.numpy().flatten(), V_eq, 'r--', linewidth=2, label='Equilibrium (V0 - μ/F)')
    ax.axhline(constants.V0, color='g', linestyle=':', alpha=0.7, label='Reference Voltage')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage (V vs Li/Li⁺)')
    ax.set_title('Voltage Profile During Phase Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# =====================================================
# STREAMLIT APP
# =====================================================

def main():
    st.set_page_config(
        page_title="LiFePO₄ Phase Decomposition PINN",
        page_icon="🔋",
        layout="wide"
    )
    
    st.title("🔋 Physics-Informed Neural Network for LiFePO₄ Phase Decomposition")
    
    st.markdown("""
    ### Simulating LiFePO₄ ↔ FePO₄ Phase Transformation with PINNs
    
    This application uses Physics-Informed Neural Networks (PINNs) to solve the 
    Cahn-Hilliard phase field model for battery electrode materials.
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Simulation Parameters")
        
        # Geometry selection
        geometry = st.selectbox(
            "Geometry",
            ["cartesian_2d", "spherical_1d"],
            format_func=lambda x: "2D Planar Electrode" if x == "cartesian_2d" else "1D Spherical Particle"
        )
        
        # Physics options
        include_voltage = st.checkbox("Include Voltage Prediction", value=True)
        include_butler_volmer = st.checkbox("Include Butler-Volmer Kinetics", 
                                          value=(geometry == "spherical_1d"))
        
        # Training parameters
        st.subheader("Training Parameters")
        epochs = st.slider("Number of Epochs", 100, 10000, 2000, 100)
        learning_rate = st.selectbox("Learning Rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4], index=2)
        
        # Material parameters
        st.subheader("Material Parameters")
        particle_radius = st.slider("Particle Radius (nm)", 10, 200, 50, 10) * 1e-9
        
        W_scale = st.slider("Phase Separation Strength (W/W₀)", 0.5, 5.0, 1.0, 0.1)
        mobility_scale = st.slider("Mobility Scale (M/M₀)", 0.1, 10.0, 1.0, 0.1)
        
        # Initial conditions
        st.subheader("Initial Conditions")
        init_type = st.selectbox("Initial Profile", 
                                ["Uniform + Noise", "Gradient", "Phase Interface"])
        
        # Experimental data upload
        st.subheader("Experimental Data")
        uploaded_file = st.file_uploader("Upload experimental data (JSON)", 
                                        type=['json'])
        
        experimental_data = None
        if uploaded_file:
            try:
                experimental_data = json.load(uploaded_file)
                st.success(f"Loaded {len(experimental_data)} data points")
            except:
                st.error("Failed to load JSON file")
        
        # Run button
        run_simulation = st.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    # Initialize constants
    constants = PhysicalConstants()
    constants.particle_radius = particle_radius
    constants.W *= W_scale
    constants.M *= mobility_scale
    
    # Main content area
    if run_simulation:
        with st.spinner("Training PINN model..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train model
            model, history = train_pinn_model(
                constants=constants,
                geometry=geometry,
                include_voltage=include_voltage,
                experimental_data=experimental_data,
                epochs=epochs,
                lr=learning_rate
            )
            
            st.success("✅ Training completed!")
        
        # Display results
        st.header("📊 Results")
        
        # Loss history
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Training Loss History")
            loss_fig = plot_loss_history(history)
            st.pyplot(loss_fig)
        
        with col2:
            st.subheader("Final Loss Values")
            loss_df = pd.DataFrame({
                'Loss Type': ['Total', 'PDE', 'BC', 'IC', 'Voltage', 'Data'],
                'Value': [
                    history['total_loss'][-1],
                    history['pde_loss'][-1],
                    history['bc_loss'][-1],
                    history['ic_loss'][-1],
                    history['voltage_loss'][-1] if include_voltage else 0,
                    history['data_loss'][-1]
                ]
            })
            st.dataframe(loss_df.style.format({'Value': '{:.2e}'}))
        
        # Concentration profiles
        st.subheader("Phase Decomposition Profiles")
        
        if geometry == "cartesian_2d":
            # Time slider for 2D visualization
            t_value = st.slider("Select time for visualization (s)", 
                              0.0, float(constants.T_max), 
                              float(constants.T_max/2), 
                              float(constants.T_max/20))
            
            profile_fig = plot_concentration_profile_2d(model, constants, t_value)
            st.pyplot(profile_fig)
            
            # Animation over time
            if st.checkbox("Show time evolution animation"):
                time_points = np.linspace(0, constants.T_max, 6)
                cols = st.columns(3)
                
                for idx, t_val in enumerate(time_points):
                    with cols[idx % 3]:
                        fig_small, ax = plt.subplots(figsize=(4, 3))
                        x = torch.linspace(0, constants.Lx, 100).reshape(-1, 1)
                        y = torch.full_like(x, constants.Ly/2)
                        t = torch.full_like(x, t_val)
                        
                        with torch.no_grad():
                            outputs = model(x, y, t)
                            c_pred = outputs['c'].numpy()
                        
                        ax.plot(x.numpy()*1e9, c_pred, 'b-', linewidth=2)
                        ax.axhline(constants.c_alpha, color='r', linestyle='--', alpha=0.5)
                        ax.axhline(constants.c_beta, color='g', linestyle='--', alpha=0.5)
                        ax.set_title(f't = {t_val:.0f} s')
                        ax.set_xlabel('x (nm)')
                        ax.set_ylabel('Li concentration')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig_small)
                        plt.close(fig_small)
        
        else:  # spherical_1d
            # Multiple time points for spherical
            time_points = st.multiselect(
                "Select time points for spherical profiles",
                options=[0.0, constants.T_max/4, constants.T_max/2, 
                       3*constants.T_max/4, constants.T_max],
                default=[0.0, constants.T_max/2, constants.T_max]
            )
            
            if time_points:
                profile_fig = plot_spherical_profile(model, constants, time_points)
                st.pyplot(profile_fig)
        
        # Voltage profile if enabled
        if include_voltage:
            st.subheader("Voltage Evolution")
            voltage_fig = plot_voltage_profile(model, constants, geometry)
            st.pyplot(voltage_fig)
            
            # Extract voltage data
            t_values = np.linspace(0, constants.T_max, 100)
            if geometry == "cartesian_2d":
                x = torch.ones((100, 1)) * constants.Lx / 2
                y = torch.ones((100, 1)) * constants.Ly / 2
                t = torch.tensor(t_values, dtype=torch.float32).reshape(-1, 1)
                outputs = model(x, y, t)
            else:
                r = torch.ones((100, 1)) * constants.particle_radius
                t = torch.tensor(t_values, dtype=torch.float32).reshape(-1, 1)
                outputs = model(r, t)
            
            V_pred = outputs['V'].detach().numpy().flatten()
            
            # Create downloadable CSV
            voltage_df = pd.DataFrame({
                'Time (s)': t_values,
                'Voltage (V)': V_pred,
                'Reference Voltage (V)': constants.V0
            })
            
            csv = voltage_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Voltage Data (CSV)",
                data=csv,
                file_name="voltage_profile.csv",
                mime="text/csv"
            )
        
        # Physics validation
        st.subheader("🧪 Physics Validation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Check mass conservation
            if geometry == "cartesian_2d":
                # Sample points at different times
                t_test = torch.tensor([0.0, constants.T_max/2, constants.T_max]).reshape(-1, 1)
                total_mass = []
                
                for t_val in t_test:
                    x = torch.rand(1000, 1) * constants.Lx
                    y = torch.rand(1000, 1) * constants.Ly
                    t_full = torch.full_like(x, t_val.item())
                    outputs = model(x, y, t_full)
                    avg_c = outputs['c'].mean().item()
                    total_mass.append(avg_c)
                
                mass_change = max(total_mass) - min(total_mass)
                st.metric("Mass Conservation Error", f"{mass_change:.2e}")
            
            else:
                st.metric("Radial Symmetry", "✓ Enforced")
        
        with col2:
            # Check phase fractions
            if geometry == "cartesian_2d":
                x = torch.rand(1000, 1) * constants.Lx
                y = torch.rand(1000, 1) * constants.Ly
                t = torch.full_like(x, constants.T_max)
                outputs = model(x, y, t)
                c = outputs['c'].detach().numpy()
                
                phase_FePO4 = np.sum(c < 0.5) / len(c)
                phase_LiFePO4 = np.sum(c >= 0.5) / len(c)
                
                st.metric("FePO₄ Fraction", f"{phase_FePO4:.3f}")
                st.metric("LiFePO₄ Fraction", f"{phase_LiFePO4:.3f}")
        
        with col3:
            # Check interface width
            if geometry == "cartesian_2d":
                # Estimate interface width from gradient
                x = torch.linspace(0, constants.Lx, 200).reshape(-1, 1)
                y = torch.full_like(x, constants.Ly/2)
                t = torch.full_like(x, constants.T_max)
                
                x.requires_grad = True
                outputs = model(x, y, t)
                c = outputs['c']
                
                c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                                         create_graph=True)[0]
                
                interface_width = (constants.c_beta - constants.c_alpha) / torch.max(torch.abs(c_x))
                st.metric("Interface Width", f"{interface_width.item()*1e9:.1f} nm")
        
        # Model download
        st.subheader("💾 Model Export")
        
        if st.button("Save PINN Model"):
            # Save model state
            model_dict = {
                'model_state': model.state_dict(),
                'constants': {
                    'W': constants.W,
                    'kappa': constants.kappa,
                    'M': constants.M,
                    'c_alpha': constants.c_alpha,
                    'c_beta': constants.c_beta,
                    'V0': constants.V0,
                    'geometry': geometry
                },
                'history': history
            }
            
            # Convert to bytes for download
            buffer = io.BytesIO()
            torch.save(model_dict, buffer)
            buffer.seek(0)
            
            st.download_button(
                label="Download Trained Model (.pth)",
                data=buffer,
                file_name=f"lifepo4_pinn_{geometry}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth",
                mime="application/octet-stream"
            )
    
    else:
        # Show instructions when not running
        st.info("👈 Configure parameters in the sidebar and click 'Run Simulation' to start.")
        
        # Display theory information
        with st.expander("📚 Theory Overview", expanded=True):
            st.markdown("""
            ### Physics of LiFePO₄ Phase Decomposition
            
            **Governing Equations:**
            
            1. **Cahn-Hilliard Phase Field Model:**
            ```
            ∂c/∂t = ∇·(M ∇μ)
            μ = ∂f/∂c - κ∇²c
            f(c) = W c²(1-c)²
            ```
            
            2. **Voltage Prediction (Nernst Equation):**
            ```
            V = V₀ - (1/F) ⟨μ⟩
            ```
            
            3. **Butler-Volmer Kinetics (for spherical particles):**
            ```
            j = i₀[exp(αFη/RT) - exp(-(1-α)Fη/RT)]
            η = V - [V₀ - (1/F)μ_surface]
            ```
            
            **PINN Implementation:**
            
            - Neural network learns concentration `c(x,t)` and chemical potential `μ(x,t)`
            - Physics enforced through automatic differentiation
            - Boundary conditions: no-flux for electrodes, Butler-Volmer for particles
            - Data assimilation from experimental measurements
            """)
        
        # Show example experimental data format
        with st.expander("📋 Example Experimental Data Format"):
            st.code("""
            [
                {
                    "type": "voltage",
                    "time": [0, 100, 200, 300, 400, 500],
                    "voltage": [3.42, 3.41, 3.40, 3.39, 3.38, 3.37]
                },
                {
                    "type": "concentration_profile",
                    "time": 300.0,
                    "x": [0, 10e-9, 20e-9, 30e-9, 40e-9, 50e-9],
                    "y": [50e-9, 50e-9, 50e-9, 50e-9, 50e-9, 50e-9],
                    "concentration": [0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
                }
            ]
            """, language="json")

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    main()
