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
import zipfile
import matplotlib as mpl
import pyvista as pv
import hashlib

# =====================================================
# Configuration & Setup for Diffusion
# =====================================================
OUTPUT_DIR_DIFF = '/tmp/pinn_solutions'
os.makedirs(OUTPUT_DIR_DIFF, exist_ok=True)

# Configure Matplotlib for Diffusion
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['xtick.major.width'] = 1.5
mpl.rcParams['ytick.major.width'] = 1.5
mpl.rcParams['figure.dpi'] = 300

# Configure logging for Diffusion
logging.basicConfig(level=logging.INFO, filename=os.path.join(OUTPUT_DIR_DIFF, 'training.log'), filemode='a')
logger_diff = logging.getLogger(__name__)

# Fixed parameters for Diffusion
C_CU_TOP = 1.59e-03
C_CU_BOTTOM = 0.0
C_NI_TOP = 0.0
C_NI_BOTTOM = 4.0e-04
Ly = 90.0
Lx = 60.0
D11 = 0.006
D12 = 0.00427
D21 = 0.003697
D22 = 0.0054
T_max = 200.0
epochs_diff = 5000
lr_diff = 1e-3

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Helper function for cache key
def get_cache_key(*args):
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()


class SmoothSigmoid(nn.Module):
    def __init__(self, slope=1.0):
        super().__init__()
        self.k = slope
        self.scale = nn.Parameter(torch.tensor(1.0))
    def forward(self, x):
        return self.scale * 1 / (1 + torch.exp(-self.k * x))


class DualScaledPINN(nn.Module):
    def __init__(self, D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni):
        super().__init__()
        self.D11 = D11
        self.D12 = D12
        self.D21 = D21
        self.D22 = D22
        self.Lx = Lx
        self.Ly = Ly
        self.T_max = T_max
        self.C_Cu = C_Cu
        self.C_Ni = C_Ni
        self.C_Cu_norm = (C_Cu - 1.5e-3) / (2.9e-3 - 1.5e-3)
        self.C_Ni_norm = (C_Ni - 4.0e-4) / (1.8e-3 - 4.0e-4)
        self.shared_net = nn.Sequential(
            nn.Linear(5, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh()
        )
        self.cu_head = nn.Sequential(
            nn.Linear(128, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        self.ni_head = nn.Sequential(
            nn.Linear(128, 1),
            SmoothSigmoid(slope=0.5),
            nn.Linear(1, 1, bias=False),
        )
        self.cu_head[2].weight.data.fill_(C_Cu)
        self.ni_head[2].weight.data.fill_(C_Ni)

    def forward(self, x, y, t):
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        C_Cu_input = torch.full_like(x, self.C_Cu_norm)
        C_Ni_input = torch.full_like(x, self.C_Ni_norm)
        inputs = torch.cat([x_norm, y_norm, t_norm, C_Cu_input, C_Ni_input], dim=1)
        features = self.shared_net(inputs)
        cu = self.cu_head(features)
        ni = self.ni_head(features)
        return torch.cat([cu, ni], dim=1)


def laplacian_diff(c, x, y):
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c),
                              create_graph=True, retain_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c),
                              create_graph=True, retain_graph=True)[0]
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x),
                               create_graph=True, retain_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y),
                               create_graph=True, retain_graph=True)[0]
    return c_xx + c_yy


def physics_loss(model, x, y, t):
    c_pred = model(x, y, t)
    c1_pred, c2_pred = c_pred[:, 0:1], c_pred[:, 1:2]
    c1_t = torch.autograd.grad(c1_pred, t, grad_outputs=torch.ones_like(c1_pred),
                               create_graph=True, retain_graph=True)[0]
    c2_t = torch.autograd.grad(c2_pred, t, grad_outputs=torch.ones_like(c2_pred),
                               create_graph=True, retain_graph=True)[0]
    lap_c1 = laplacian_diff(c1_pred, x, y)
    lap_c2 = laplacian_diff(c2_pred, x, y)
    residual1 = c1_t - (model.D11 * lap_c1 + model.D12 * lap_c2)
    residual2 = c2_t - (model.D21 * lap_c1 + model.D22 * lap_c2)
    return torch.mean(residual1**2 + residual2**2)


def boundary_loss_bottom(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.zeros(num, 1, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_BOTTOM)**2) +
            torch.mean((c_pred[:, 1] - C_NI_BOTTOM)**2))


def boundary_loss_top(model):
    num = 200
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.full((num, 1), model.Ly, requires_grad=True)
    t = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_pred = model(x, y, t)
    return (torch.mean((c_pred[:, 0] - C_CU_TOP)**2) +
            torch.mean((c_pred[:, 1] - C_NI_TOP)**2))


def boundary_loss_sides(model):
    num = 200
    x_left = torch.zeros(num, 1, dtype=torch.float32, requires_grad=True)
    y_left = torch.rand(num, 1, requires_grad=True) * model.Ly
    t_left = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_left = model(x_left, y_left, t_left)
    x_right = torch.full((num, 1), float(model.Lx), dtype=torch.float32, requires_grad=True)
    y_right = torch.rand(num, 1, requires_grad=True) * model.Ly
    t_right = torch.rand(num, 1, requires_grad=True) * model.T_max
    c_right = model(x_right, y_right, t_right)
    try:
        grad_cu_x_left = torch.autograd.grad(
            c_left[:, 0], x_left,
            grad_outputs=torch.ones_like(c_left[:, 0]),
            create_graph=True, retain_graph=True
        )[0]
        grad_ni_x_left = torch.autograd.grad(
            c_left[:, 1], x_left,
            grad_outputs=torch.ones_like(c_left[:, 1]),
            create_graph=True, retain_graph=True
        )[0]
        grad_cu_x_right = torch.autograd.grad(
            c_right[:, 0], x_right,
            grad_outputs=torch.ones_like(c_right[:, 0]),
            create_graph=True, retain_graph=True
        )[0]
        grad_ni_x_right = torch.autograd.grad(
            c_right[:, 1], x_right,
            grad_outputs=torch.ones_like(c_right[:, 1]),
            create_graph=True, retain_graph=True
        )[0]
        grad_cu_x_left = grad_cu_x_left if grad_cu_x_left is not None else torch.zeros_like(c_left[:, 0])
        grad_ni_x_left = grad_ni_x_left if grad_ni_x_left is not None else torch.zeros_like(c_left[:, 1])
        grad_cu_x_right = grad_cu_x_right if grad_cu_x_right is not None else torch.zeros_like(c_right[:, 0])
        grad_ni_x_right = grad_ni_x_right if grad_ni_x_right is not None else torch.zeros_like(c_right[:, 1])
        return (torch.mean(grad_cu_x_left**2) +
                torch.mean(grad_ni_x_left**2) +
                torch.mean(grad_cu_x_right**2) +
                torch.mean(grad_ni_x_right**2))
    except RuntimeError as e:
        logger_diff.error(f"Gradient computation failed in boundary_loss_sides: {str(e)}")
        st.error(f"Gradient computation failed: {str(e)}")
        return torch.tensor(1e-6, requires_grad=True)


def initial_loss(model):
    num = 500
    x = torch.rand(num, 1, requires_grad=True) * model.Lx
    y = torch.rand(num, 1, requires_grad=True) * model.Ly
    t = torch.zeros(num, 1, requires_grad=True)
    return torch.mean(model(x, y, t)**2)


def validate_boundary_conditions(solution, tolerance=1e-6):
    results = {
        'top_bc_cu': True,
        'top_bc_ni': True,
        'bottom_bc_cu': True,
        'bottom_bc_ni': True,
        'left_flux_cu': True,
        'left_flux_ni': True,
        'right_flux_cu': True,
        'right_flux_ni': True,
        'details': []
    }
    t_idx = -1
    c1 = solution['c1_preds'][t_idx]
    c2 = solution['c2_preds'][t_idx]
    top_cu_mean = np.mean(c1[:, -1])
    top_ni_mean = np.mean(c2[:, -1])
    if abs(top_cu_mean - C_CU_TOP) > tolerance:
        results['top_bc_cu'] = False
        results['details'].append(f"Top Cu: {top_cu_mean:.2e} != {C_CU_TOP:.2e}")
    if abs(top_ni_mean - C_NI_TOP) > tolerance:
        results['top_bc_ni'] = False
        results['details'].append(f"Top Ni: {top_ni_mean:.2e} != {C_NI_TOP:.2e}")
    bottom_cu_mean = np.mean(c1[:, 0])
    bottom_ni_mean = np.mean(c2[:, 0])
    if abs(bottom_cu_mean - C_CU_BOTTOM) > tolerance:
        results['bottom_bc_cu'] = False
        results['details'].append(f"Bottom Cu: {bottom_cu_mean:.2e} != {C_CU_BOTTOM:.2e}")
    if abs(bottom_ni_mean - C_NI_BOTTOM) > tolerance:
        results['bottom_bc_ni'] = False
        results['details'].append(f"Bottom Ni: {bottom_ni_mean:.2e} != {C_NI_BOTTOM:.2e}")
    left_flux_cu = np.mean(np.abs(c1[1, :] - c1[0, :]))
    left_flux_ni = np.mean(np.abs(c2[1, :] - c2[0, :]))
    right_flux_cu = np.mean(np.abs(c1[-1, :] - c1[-2, :]))
    right_flux_ni = np.mean(np.abs(c2[-1, :] - c2[-2, :]))
    if left_flux_cu > tolerance:
        results['left_flux_cu'] = False
        results['details'].append(f"Left flux Cu: {left_flux_cu:.2e}")
    if left_flux_ni > tolerance:
        results['left_flux_ni'] = False
        results['details'].append(f"Left flux Ni: {left_flux_ni:.2e}")
    if right_flux_cu > tolerance:
        results['right_flux_cu'] = False
        results['details'].append(f"Right flux Cu: {right_flux_cu:.2e}")
    if right_flux_ni > tolerance:
        results['right_flux_ni'] = False
        results['details'].append(f"Right flux Ni: {right_flux_ni:.2e}")
    results['valid'] = all([
        results['top_bc_cu'], results['top_bc_ni'],
        results['bottom_bc_cu'], results['bottom_bc_ni'],
        results['left_flux_cu'], results['left_flux_ni'],
        results['right_flux_cu'], results['right_flux_ni']
    ])
    return results


@st.cache_data(ttl=3600, show_spinner=False)
def plot_losses_diff(loss_history, output_dir, _hash):
    epochs = np.array(loss_history['epochs'])
    total_loss = np.array(loss_history['total'])
    physics_loss = np.array(loss_history['physics'])
    bottom_loss = np.array(loss_history['bottom'])
    top_loss = np.array(loss_history['top'])
    sides_loss = np.array(loss_history['sides'])
    initial_loss = np.array(loss_history['initial'])
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, total_loss, label='Total Loss', linewidth=2, color='black')
    plt.plot(epochs, physics_loss, label='Physics Loss', linewidth=1.5, linestyle='--', color='blue')
    plt.plot(epochs, bottom_loss, label='Bottom Boundary Loss', linewidth=1.5, linestyle='-.', color='red')
    plt.plot(epochs, top_loss, label='Top Boundary Loss', linewidth=1.5, linestyle=':', color='green')
    plt.plot(epochs, sides_loss, label='Sides Boundary Loss', linewidth=1.5, linestyle='-', color='purple')
    plt.plot(epochs, initial_loss, label='Initial Condition Loss', linewidth=1.5, linestyle='--', color='orange')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss for Ly = {Ly:.1f} μm')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'loss_plot_ly_{Ly:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger_diff.info(f"Saved loss plot to {plot_filename}")
    return plot_filename


@st.cache_data(ttl=3600, show_spinner=False)
def plot_2d_profiles(solution, time_idx, output_dir, _hash):
    t_val = solution['times'][time_idx]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(solution['c1_preds'][time_idx], origin='lower',
                     extent=[0, Lx, 0, Ly], cmap='viridis',
                     vmin=0, vmax=C_CU_TOP)
    plt.title(f'Cu Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im1, label='Cu Conc. (mol/cc)', format='%.1e')
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(solution['c2_preds'][time_idx], origin='lower',
                     extent=[0, Lx, 0, Ly], cmap='magma',
                     vmin=0, vmax=C_NI_BOTTOM)
    plt.title(f'Ni Concentration (t={t_val:.1f} s)')
    plt.xlabel('x (μm)')
    plt.ylabel('y (μm)')
    plt.grid(True, alpha=0.3)
    plt.colorbar(im2, label='Ni Conc. (mol/cc)', format='%.1e')
    plt.suptitle(f'2D Profiles (Ly={Ly:.0f} μm)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plot_filename = os.path.join(output_dir, f'profile_ly_{Ly:.1f}_t_{t_val:.1f}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    logger_diff.info(f"Saved profile plot to {plot_filename}")
    return plot_filename


@st.cache_resource(ttl=3600, show_spinner=False)
def train_model_diff(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni, epochs, lr, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    logger_diff.info(f"Starting training with Ly={Ly}, C_Cu={C_Cu}, C_Ni={C_Ni}, epochs={epochs}, lr={lr}")
    model = DualScaledPINN(D11, D12, D21, D22, Lx, Ly, T_max, C_Cu, C_Ni)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    x_pde = torch.rand(1000, 1, requires_grad=True) * Lx
    y_pde = torch.rand(1000, 1, requires_grad=True) * Ly
    t_pde = torch.rand(1000, 1, requires_grad=True) * T_max
    loss_history = {
        'epochs': [],
        'total': [],
        'physics': [],
        'bottom': [],
        'top': [],
        'sides': [],
        'initial': []
    }
    progress = st.progress(0)
    status_text = st.empty()
    for epoch in range(epochs):
        optimizer.zero_grad()
        phys_loss = physics_loss(model, x_pde, y_pde, t_pde)
        bot_loss = boundary_loss_bottom(model)
        top_loss = boundary_loss_top(model)
        side_loss = boundary_loss_sides(model)
        init_loss = initial_loss(model)
        loss = (10 * phys_loss + 100 * bot_loss + 100 * top_loss +
                100 * side_loss + 100 * init_loss)
        try:
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        except RuntimeError as e:
            logger_diff.error(f"Backward pass failed at epoch {epoch + 1}: {str(e)}")
            st.error(f"Training failed at epoch {epoch + 1}: {str(e)}")
            return None, None
        if (epoch + 1) % 100 == 0:
            loss_history['epochs'].append(epoch + 1)
            loss_history['total'].append(loss.item())
            loss_history['physics'].append(10 * phys_loss.item())
            loss_history['bottom'].append(100 * bot_loss.item())
            loss_history['top'].append(100 * top_loss.item())
            loss_history['sides'].append(100 * side_loss.item())
            loss_history['initial'].append(100 * init_loss.item())
            progress.progress((epoch + 1) / epochs)
            status_text.text(
                f"Epoch {epoch + 1}/{epochs}, Total Loss: {loss.item():.6f}, "
                f"Physics: {10 * phys_loss.item():.6f}, Bottom: {100 * bot_loss.item():.6f}, "
                f"Top: {100 * top_loss.item():.6f}, Sides: {100 * side_loss.item():.6f}, "
                f"Initial: {100 * init_loss.item():.6f}"
            )
    progress.progress(1.0)
    status_text.text("Training completed!")
    logger_diff.info("Training completed successfully")
    return model, loss_history


@st.cache_data(ttl=3600, show_spinner=False)
def evaluate_model_diff(_model, times, Lx, Ly, D11, D12, D21, D22, _hash):
    x = torch.linspace(0, Lx, 50, requires_grad=False)
    y = torch.linspace(0, Ly, 50, requires_grad=False)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    c1_preds, c2_preds = [], []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val, requires_grad=False)
        c_pred = _model(X.reshape(-1,1), Y.reshape(-1,1), t)
        try:
            c1 = c_pred[:,0].detach().numpy().reshape(50,50).T
            c2 = c_pred[:,1].detach().numpy().reshape(50,50).T
        except RuntimeError as e:
            logger_diff.error(f"Failed to convert concentration predictions to NumPy: {str(e)}")
            raise e
        c1_preds.append(c1)
        c2_preds.append(c2)
    return X.numpy(), Y.numpy(), c1_preds, c2_preds


@st.cache_data(ttl=3600, show_spinner=False)
def generate_and_save_solution_diff(_model, times, param_set, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    if _model is None:
        logger_diff.error("Model is None, cannot generate solution")
        return None, None
    try:
        X, Y, c1_preds, c2_preds = evaluate_model_diff(
            _model, times, param_set['Lx'], param_set['Ly'],
            param_set['D11'], param_set['D12'], param_set['D21'], param_set['D22'], _hash
        )
    except RuntimeError as e:
        logger_diff.error(f"evaluate_model failed: {str(e)}")
        st.error(f"evaluate_model failed: {str(e)}")
        return None, None
    solution = {
        'params': param_set,
        'X': X,
        'Y': Y,
        'c1_preds': c1_preds,
        'c2_preds': c2_preds,
        'times': times,
        'loss_history': {},
        'orientation_note': 'c1_preds and c2_preds are arrays of shape (50,50) where rows (i) correspond to y-coordinates and columns (j) correspond to x-coordinates for matplotlib.'
    }
    solution_filename = os.path.join(output_dir,
        f"solution_crossdiffusion_ly_{param_set['Ly']:.1f}_tmax_{param_set['t_max']:.1f}.pkl")
    try:
        with open(solution_filename, 'wb') as f:
            pickle.dump(solution, f)
        logger_diff.info(f"Saved solution to {solution_filename}")
    except Exception as e:
        logger_diff.error(f"Failed to save solution: {str(e)}")
        st.error(f"Failed to save solution: {str(e)}")
        return None, None
    return solution_filename, solution


@st.cache_data(ttl=3600, show_spinner=False)
def generate_vts_time_series(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    times = solution['times']
    vts_files = []
    nx, ny = 50, 50
    for t_idx, t_val in enumerate(times):
        c1_xy = solution['c1_preds'][t_idx].T
        c2_xy = solution['c2_preds'][t_idx].T
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.zeros((nx, ny))
        grid = pv.StructuredGrid()
        X, Y = np.meshgrid(x, y, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
        grid.points = points
        grid.dimensions = (nx, ny, 1)
        grid.point_data['Cu_Concentration'] = c1_xy.ravel()
        grid.point_data['Ni_Concentration'] = c2_xy.ravel()
        vts_filename = os.path.join(output_dir,
            f'concentration_ly_{Ly:.1f}_t_{t_val:.1f}.vts')
        try:
            grid.save(vts_filename)
            vts_files.append((t_val, vts_filename))
            logger_diff.info(f"Saved VTS file to {vts_filename}")
        except Exception as e:
            logger_diff.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
            st.error(f"Failed to save VTS file for t={t_val:.1f}: {str(e)}")
    pvd_filename = os.path.join(output_dir,
        f'concentration_time_series_ly_{Ly:.1f}.pvd')
    try:
        pvd_content = ['<?xml version="1.0"?>']
        pvd_content.append('<VTKFile type="Collection" version="0.1">')
        pvd_content.append(' <Collection>')
        for t_val, vts_file in vts_files:
            relative_path = os.path.basename(vts_file)
            pvd_content.append(f' <DataSet timestep="{t_val}" group="" part="0" file="{relative_path}"/>')
        pvd_content.append(' </Collection>')
        pvd_content.append('</VTKFile>')
        with open(pvd_filename, 'w') as f:
            f.write('\n'.join(pvd_content))
        logger_diff.info(f"Saved PVD collection file to {pvd_filename}")
    except Exception as e:
        logger_diff.error(f"Failed to create PVD file: {str(e)}")
        st.error(f"Failed to create PVD file: {str(e)}")
        pvd_filename = None
    return vts_files, pvd_filename


@st.cache_data(ttl=3600, show_spinner=False)
def generate_vtu_time_series(solution, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    Lx = solution['params']['Lx']
    Ly = solution['params']['Ly']
    times = solution['times']
    nx, ny = 50, 50
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.zeros((nx, ny))
    X, Y = np.meshgrid(x, y, indexing='ij')
    points = np.stack([X.ravel(), Y.ravel(), z.ravel()], axis=1)
    cells = []
    cell_types = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            idx = i + j * nx
            cell = [4, idx, idx + 1, idx + nx + 1, idx + nx]
            cells.extend(cell)
            cell_types.append(pv.CellType.QUAD)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    for t_idx, t_val in enumerate(times):
        c1_xy = solution['c1_preds'][t_idx].T
        c2_xy = solution['c2_preds'][t_idx].T
        grid.point_data[f'Cu_Concentration_t{t_val:.1f}'] = c1_xy.ravel()
        grid.point_data[f'Ni_Concentration_t{t_val:.1f}'] = c2_xy.ravel()
    vtu_filename = os.path.join(output_dir,
        f'concentration_time_series_ly_{Ly:.1f}.vtu')
    try:
        grid.save(vtu_filename)
        logger_diff.info(f"Saved VTU file to {vtu_filename}")
    except Exception as e:
        logger_diff.error(f"Failed to save VTU file: {str(e)}")
        st.error(f"Failed to save VTU file: {str(e)}")
        return None
    return vtu_filename


@st.cache_data(ttl=3600, show_spinner=False)
def create_zip_file(_files, output_dir, _hash):
    os.makedirs(output_dir, exist_ok=True)
    zip_buffer = io.BytesIO()
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in _files:
                if os.path.exists(file_path):
                    zip_file.write(file_path, os.path.basename(file_path))
                else:
                    logger_diff.warning(f"File not found for zipping: {file_path}")
        zip_filename = os.path.join(output_dir, f'pinn_solutions_crossdiffusion_ly_{Ly:.1f}.zip')
        with open(zip_filename, 'wb') as f:
            f.write(zip_buffer.getvalue())
        logger_diff.info(f"Created ZIP file: {zip_filename}")
        return zip_filename
    except Exception as e:
        logger_diff.error(f"Failed to create ZIP file: {str(e)}")
        st.error(f"Failed to create ZIP file: {str(e)}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def get_file_bytes(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return f.read()
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def train_and_generate_solution_diff(_model, loss_history, output_dir, _hash_key):
    os.makedirs(output_dir, exist_ok=True)
    if _model is None or loss_history is None:
        return None, None
    times = np.linspace(0, T_max, 50)
    param_set = {
        'D11': D11, 'D12': D12, 'D21': D21, 'D22': D22,
        'Lx': Lx, 'Ly': Ly, 't_max': T_max,
        'C_Cu': C_CU_TOP, 'C_Ni': C_NI_BOTTOM,
        'epochs': epochs_diff
    }
    solution_filename, solution = generate_and_save_solution_diff(
        _model, times, param_set, output_dir, _hash_key
    )
    if solution is None:
        return None, None
    solution['loss_history'] = loss_history
    loss_plot_filename = plot_losses_diff(loss_history, output_dir, _hash_key)
    profile_plot_filename = plot_2d_profiles(solution, -1, output_dir, _hash_key)
    vts_files, pvd_file = generate_vts_time_series(solution, output_dir, _hash_key)
    vtu_file = generate_vtu_time_series(solution, output_dir, _hash_key)
    return solution, {
        'solution_file': solution_filename,
        'loss_plot': loss_plot_filename,
        'profile_plot': profile_plot_filename,
        'vts_files': vts_files,
        'pvd_file': pvd_file,
        'vtu_file': vtu_file
    }


def initialize_session_state_diff():
    if 'training_complete_diff' not in st.session_state:
        st.session_state.training_complete_diff = False
    if 'solution_data_diff' not in st.session_state:
        st.session_state.solution_data_diff = None
    if 'file_data_diff' not in st.session_state:
        st.session_state.file_data_diff = {}
    if 'model_diff' not in st.session_state:
        st.session_state.model_diff = None
    if 'current_hash_diff' not in st.session_state:
        st.session_state.current_hash_diff = None


def store_solution_in_session_diff(_hash_key, solution, file_info, model):
    st.session_state.training_complete_diff = True
    st.session_state.solution_data_diff = solution
    st.session_state.file_data_diff = file_info
    st.session_state.model_diff = model
    st.session_state.current_hash_diff = _hash_key


def main_diffusion():
    st.title("2D PINN Simulation: Cu-Ni Cross-Diffusion for Liquid Solder Height = 90 micrometers")
    initialize_session_state_diff()
    current_hash = get_cache_key(Ly, C_CU_TOP, C_NI_BOTTOM, epochs_diff, lr_diff)
    if st.session_state.training_complete_diff and st.session_state.current_hash_diff == current_hash:
        solution = st.session_state.solution_data_diff
        file_info = st.session_state.file_data_diff
        model = st.session_state.model_diff
        st.info("Displaying cached results.")
    else:
        solution = None
        file_info = {}
        model = None
        st.warning("No results available. Click 'Run Simulation' to generate results.")
    if st.button("Run Simulation"):
        try:
            with st.spinner("Running simulation..."):
                model, loss_history = train_model_diff(
                    D11, D12, D21, D22, Lx, Ly, T_max, C_CU_TOP, C_NI_BOTTOM, epochs_diff, lr_diff, OUTPUT_DIR_DIFF, current_hash
                )
                if model is None or loss_history is None:
                    st.error("Simulation failed!")
                    return
                solution, file_info = train_and_generate_solution_diff(
                    model, loss_history, OUTPUT_DIR_DIFF, current_hash
                )
                if solution is None:
                    st.error("Solution generation failed!")
                    return
                store_solution_in_session_diff(current_hash, solution, file_info, model)
                st.success("Simulation completed successfully!")
        except Exception as e:
            logger_diff.error(f"Simulation failed: {str(e)}")
            st.error(f"Simulation failed: {str(e)}")
            return
    if solution and file_info:
        with st.expander("Training Logs", expanded=False):
            log_file = os.path.join(OUTPUT_DIR_DIFF, 'training.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    st.text(f.read())
        st.subheader("Training Loss")
        st.image(file_info['loss_plot'])
        st.subheader("Boundary Condition Validation")
        bc_results = validate_boundary_conditions(solution)
        st.metric("Boundary Conditions", "✓" if bc_results['valid'] else "✗",
                  f"{len(bc_results['details'])} issues")
        with st.expander("Boundary Condition Details"):
            for issue in bc_results['details']:
                st.write(f"• {issue}")
        st.subheader("2D Concentration Profiles (Final Time Step)")
        st.image(file_info['profile_plot'])
        st.subheader("Download Files")
        solution_filename = file_info.get('solution_file')
        if solution_filename and os.path.exists(solution_filename):
            solution_data = get_file_bytes(solution_filename)
            if solution_data:
                st.download_button(
                    label="Download Solution (.pkl)",
                    data=solution_data,
                    file_name=os.path.basename(solution_filename),
                    mime="application/octet-stream"
                )
        for file_type, file_path in [
            ("Loss Plot", file_info['loss_plot']),
            ("2D Profile Plot", file_info['profile_plot'])
        ]:
            if os.path.exists(file_path):
                file_data = get_file_bytes(file_path)
                if file_data:
                    st.download_button(
                        label=f"Download {file_type} (.png)",
                        data=file_data,
                        file_name=os.path.basename(file_path),
                        mime="image/png"
                    )
        st.subheader("Download Time Series Files")
        if file_info.get('pvd_file') and os.path.exists(file_info['pvd_file']):
            pvd_data = get_file_bytes(file_info['pvd_file'])
            if pvd_data:
                st.download_button(
                    label="Download VTS Time Series (.pvd + .vts)",
                    data=pvd_data,
                    file_name=os.path.basename(file_info['pvd_file']),
                    mime="application/xml",
                    help="Download the PVD collection file. Keep all .vts files in the same folder."
                )
        if file_info.get('vtu_file') and os.path.exists(file_info['vtu_file']):
            vtu_data = get_file_bytes(file_info['vtu_file'])
            if vtu_data:
                st.download_button(
                    label="Download VTU Time Series (.vtu)",
                    data=vtu_data,
                    file_name=os.path.basename(file_info['vtu_file']),
                    mime="application/xml",
                    help="Single VTU file with all timesteps."
                )
        st.subheader("Download Individual Time Steps")
        for t_val, vts_file in file_info.get('vts_files', []):
            if os.path.exists(vts_file):
                vts_data = get_file_bytes(vts_file)
                if vts_data:
                    st.download_button(
                        label=f"Download Time = {t_val:.1f} s (.vts)",
                        data=vts_data,
                        file_name=os.path.basename(vts_file),
                        mime="application/xml"
                    )
        st.subheader("Download All Files as ZIP")
        if st.button("Generate ZIP File"):
            with st.spinner("Creating ZIP file..."):
                files_to_zip = [
                    file_info['loss_plot'],
                    file_info['profile_plot']
                ]
                if solution_filename:
                    files_to_zip.append(solution_filename)
                for _, vts_file in file_info.get('vts_files', []):
                    files_to_zip.append(vts_file)
                if file_info.get('pvd_file'):
                    files_to_zip.append(file_info['pvd_file'])
                if file_info.get('vtu_file'):
                    files_to_zip.append(file_info['vtu_file'])
                zip_filename = create_zip_file(files_to_zip, OUTPUT_DIR_DIFF, (Ly,))
                if zip_filename and os.path.exists(zip_filename):
                    zip_data = get_file_bytes(zip_filename)
                    if zip_data:
                        st.download_button(
                            label="Download All Files (.zip)",
                            data=zip_data,
                            file_name=os.path.basename(zip_filename),
                            mime="application/zip"
                        )


# =====================================================
# Configuration & Setup for Phase Field
# =====================================================
OUTPUT_DIR_PHASE = '/tmp/pinn_phase_field'
os.makedirs(OUTPUT_DIR_PHASE, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR_PHASE, 'pinn_training.log')),
        logging.StreamHandler()
    ]
)
logger_phase = logging.getLogger(__name__)

DEFAULT_PARAMS_PHASE = {
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
    'epochs': 1000,
    'lr': 1e-3,
    'pde_points': 2000,
    'ic_points': 500,
}


# =====================================================
# Numba-accelerated Phase Field Simulation Functions
# =====================================================
@njit(fastmath=True, cache=True)
def double_well_energy(c, A, B, C):
    return A * c**2 + B * c**3 + C * c**4

@njit(fastmath=True, cache=True)
def chemical_potential_phase(c, A, B, C):
    return 2.0 * A * c + 3.0 * B * c**2 + 4.0 * C * c**3

@njit(fastmath=True, parallel=False)
def compute_laplacian_phase(field, dx):
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

@njit(fastmath=True, parallel=False)
def compute_gradient_x_phase(field, dx):
    nx, ny = field.shape
    grad_x = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            ip1 = (i + 1) % nx
            im1 = (i - 1) % nx
            grad_x[i, j] = (field[ip1, j] - field[im1, j]) / (2.0 * dx)
    return grad_x

@njit(fastmath=True, parallel=False)
def compute_gradient_y_phase(field, dx):
    nx, ny = field.shape
    grad_y = np.zeros_like(field)
    for i in prange(nx):
        for j in prange(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            grad_y[i, j] = (field[i, jp1] - field[i, jm1]) / (2.0 * dx)
    return grad_y

@njit(fastmath=True, parallel=False)
def update_concentration_phase(c, dt, dx, kappa, M, A, B, C):
    nx, ny = c.shape
    lap_c = compute_laplacian_phase(c, dx)
    mu = chemical_potential_phase(c, A, B, C) - kappa * lap_c
    mu_x = compute_gradient_x_phase(mu, dx)
    mu_y = compute_gradient_y_phase(mu, dx)
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

    def set_parameters_phase(self, W=None, kappa=None, M=None, A=None, B=None, C=None):
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

    def initialize_random_phase(self, c0=0.5, noise_amplitude=0.01):
        self.c = c0 + noise_amplitude * (2.0 * np.random.random((self.nx, self.ny)) - 1.0)
        self.time = 0.0
        self.step = 0
        self.clear_history_phase()

    def initialize_seed_phase(self, c0=0.3, seed_value=0.7, radius=15):
        self.c = c0 * np.ones((self.nx, self.ny))
        center_x, center_y = self.nx // 2, self.ny // 2
        for i in range(self.nx):
            for j in range(self.ny):
                if (i - center_x)**2 + (j - center_y)**2 < radius**2:
                    self.c[i, j] = seed_value
        self.time = 0.0
        self.step = 0
        self.clear_history_phase()

    def clear_history_phase(self):
        self.history = {
            'time': [],
            'mean': [],
            'std': [],
            'phase_high': [],
            'phase_low': []
        }
        self.update_history_phase()

    def update_history_phase(self):
        self.history['time'].append(self.time)
        self.history['mean'].append(np.mean(self.c))
        self.history['std'].append(np.std(self.c))
        self.history['phase_high'].append(np.sum(self.c > 0.5) / (self.nx * self.ny))
        self.history['phase_low'].append(np.sum(self.c < 0.5) / (self.nx * self.ny))

    def run_step_phase(self):
        self.c = update_concentration_phase(
            self.c, self.dt, self.dx,
            self.kappa, self.M,
            self.A, self.B, self.C
        )
        self.time += self.dt
        self.step += 1
        self.update_history_phase()

    def run_steps_phase(self, n_steps):
        for _ in range(n_steps):
            self.run_step_phase()

    def compute_free_energy_density_phase(self):
        energy = np.zeros_like(self.c)
        for i in range(self.nx):
            for j in range(self.ny):
                energy[i, j] = double_well_energy(self.c[i, j], self.A, self.B, self.C)
        return energy

    def get_statistics_phase(self):
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
# PINN Model Definition for Phase Field
# =====================================================
class CahnHilliardPINN(nn.Module):
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
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm = x / self.Lx
        y_norm = y / self.Ly
        t_norm = t / self.T_max
        x_sin = torch.sin(2 * np.pi * x_norm)
        x_cos = torch.cos(2 * np.pi * x_norm)
        y_sin = torch.sin(2 * np.pi * y_norm)
        y_cos = torch.cos(2 * np.pi * y_norm)
        inputs = torch.cat([x_sin, x_cos, y_sin, y_cos, t_norm], dim=1).float()
        c = self.net(inputs)
        return c

    def chemical_potential_phase(self, c: torch.Tensor) -> torch.Tensor:
        return 2.0 * self.A * c + 3.0 * self.B * c**2 + 4.0 * self.C * c**3


# =====================================================
# PINN Physics Utilities for Phase Field
# =====================================================
def laplacian_phase(u: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
    return u_xx + u_yy


def pde_residual_phase(model: CahnHilliardPINN,
                      x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    c = model(x, y, t)
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True, retain_graph=True)[0]
    lap_c = laplacian_phase(c, x, y)
    mu = model.chemical_potential_phase(c) - model.kappa * lap_c
    lap_mu = laplacian_phase(mu, x, y)
    residual = c_t - model.M * lap_mu
    return residual


def initial_condition_loss_phase(model: CahnHilliardPINN,
                                x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    noise_x = torch.sin(2 * np.pi * x / model.Lx) * torch.cos(4 * np.pi * y / model.Ly)
    noise_y = torch.cos(4 * np.pi * x / model.Lx) * torch.sin(2 * np.pi * y / model.Ly)
    smooth_noise = 0.5 * (noise_x + noise_y)
    c_target = model.c0 + model.noise_amp * smooth_noise
    c_pred = model(x, y, torch.zeros_like(x))
    return torch.mean((c_pred - c_target)**2)


# =====================================================
# PINN Training Functions for Phase Field
# =====================================================
def generate_collocation_points_phase(pde_points: int, ic_points: int,
                                      Lx: float, Ly: float, T_max: float) -> Dict[str, torch.Tensor]:
    x_pde = torch.rand(pde_points, 1, requires_grad=True) * Lx
    y_pde = torch.rand(pde_points, 1, requires_grad=True) * Ly
    t_pde = torch.rand(pde_points, 1, requires_grad=True) * T_max
    x_ic = torch.rand(ic_points, 1, requires_grad=True) * Lx
    y_ic = torch.rand(ic_points, 1, requires_grad=True) * Ly
    return {
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_ic': x_ic, 'y_ic': y_ic
    }


@st.cache_resource
def train_pinn_model_phase(params: Dict,
                           progress_callback=None,
                           status_callback=None) -> Tuple[Optional[CahnHilliardPINN], Dict]:
    logger_phase.info(f"Starting PINN training with parameters: {params}")
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
    A = W
    B = -2.0 * W
    C = W
    model = CahnHilliardPINN(
        Lx=Lx, Ly=Ly, T_max=T_max,
        A=A, B=B, C=C,
        kappa=kappa, M=M,
        c0=c0, noise_amp=noise_amp
    )
    points = generate_collocation_points_phase(pde_points, ic_points, Lx, Ly, T_max)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)
    loss_history = {
        'epochs': [],
        'total': [],
        'pde': [],
        'ic': []
    }
    batch_size = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        try:
            pde_loss = 0.0
            num_pde_batches = (pde_points + batch_size - 1) // batch_size
            for i in range(0, pde_points, batch_size):
                end = min(i + batch_size, pde_points)
                x_batch = points['x_pde'][i:end]
                y_batch = points['y_pde'][i:end]
                t_batch = points['t_pde'][i:end]
                residual = pde_residual_phase(model, x_batch, y_batch, t_batch)
                pde_loss += torch.mean(residual**2) / num_pde_batches
            ic_loss = 0.0
            num_ic_batches = (ic_points + batch_size - 1) // batch_size
            for i in range(0, ic_points, batch_size):
                end = min(i + batch_size, ic_points)
                x_batch = points['x_ic'][i:end]
                y_batch = points['y_ic'][i:end]
                ic_loss += initial_condition_loss_phase(model, x_batch, y_batch) / num_ic_batches
            total_loss = pde_loss + 10.0 * ic_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(total_loss)
        except RuntimeError as e:
            logger_phase.error(f"Training failed at epoch {epoch}: {str(e)}")
            if "out of memory" in str(e):
                st.error("Out of memory! Reduce points or epochs further.")
            else:
                st.error(f"Training error: {str(e)}")
            return None, {}
        except MemoryError as e:
            logger_phase.error(f"Memory error at epoch {epoch}: {str(e)}")
            st.error("Memory error during training!")
            return None, {}
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
    logger_phase.info("PINN training completed successfully")
    return model, loss_history


# =====================================================
# PINN Evaluation & Visualization for Phase Field
# =====================================================
def evaluate_pinn_solution_phase(model: CahnHilliardPINN,
                                times: np.ndarray,
                                nx: int = 128, ny: int = 128) -> Dict:
    Lx, Ly = model.Lx, model.Ly
    x = torch.linspace(0, Lx, nx)
    y = torch.linspace(0, Ly, ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    c_solutions = []
    free_energy_solutions = []
    for t_val in times:
        t = torch.full((X.numel(), 1), t_val, dtype=torch.float32)
        with torch.no_grad():
            c_pred = model(X.reshape(-1, 1), Y.reshape(-1, 1), t)
        c = c_pred.reshape(nx, ny).cpu().numpy()
        c_solutions.append(c)
        energy = model.A * c**2 + model.B * c**3 + model.C * c**4
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


def plot_pinn_losses_phase(loss_history: Dict, output_dir: str) -> str:
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


def plot_pinn_concentration_phase(solution: Dict, time_idx: int, output_dir: str) -> str:
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


def plot_pinn_free_energy_phase(solution: Dict, time_idx: int, output_dir: str) -> str:
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


def plot_pinn_histogram_phase(solution: Dict, time_idx: int, output_dir: str) -> str:
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
# Streamlit PINN Integration for Phase Field
# =====================================================
def add_pinn_section_phase():
    st.header("🧠 Physics-Informed Neural Network (PINN)")
    st.markdown("""
    This section uses a **mesh-free neural network** to solve the Cahn-Hilliard equation directly from physics.
    No numerical discretization—just automatic differentiation and optimization!
    """)
    with st.sidebar:
        st.divider()
        st.subheader("🧠 PINN Parameters")
        pinn_epochs = st.number_input("Training epochs", 1000, 10000, 3000, 100)
        pinn_lr = st.number_input("Learning rate", 1e-5, 1e-2, 1e-3, format="%.1e")
        pde_points = st.number_input("PDE collocation points", 1000, 20000, 8000, 1000)
        ic_points = st.number_input("IC points", 500, 5000, 2000, 500)
        st.info("💡 **Tip**: Start with fewer points for faster testing, then increase for accuracy.")
    if 'sim_phase' in st.session_state:
        sim = st.session_state.sim_phase
        current_params = {
            'Lx': float(sim.dx * sim.nx),
            'Ly': float(sim.dx * sim.ny),
            'T_max': DEFAULT_PARAMS_PHASE['T_max'],
            'W': sim.W,
            'kappa': sim.kappa,
            'M': sim.M,
            'c0': 0.5,
            'noise_amp': 0.05,
            'epochs': pinn_epochs,
            'lr': pinn_lr,
            'pde_points': pde_points,
            'ic_points': ic_points,
            'nx': DEFAULT_PARAMS_PHASE['nx'],
            'ny': DEFAULT_PARAMS_PHASE['ny']
        }
    else:
        current_params = DEFAULT_PARAMS_PHASE.copy()
        current_params.update({
            'epochs': pinn_epochs,
            'lr': pinn_lr,
            'pde_points': pde_points,
            'ic_points': ic_points
        })
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
                model, loss_history = train_pinn_model_phase(
                    current_params,
                    progress_callback=update_progress,
                    status_callback=update_status
                )
                training_time = time.time() - start_time
                if model is None:
                    st.error("PINN training failed!")
                    return
                st.session_state.pinn_model_phase = model
                st.session_state.pinn_loss_history_phase = loss_history
                st.session_state.pinn_params_phase = current_params
                st.session_state.pinn_training_time_phase = training_time
                st.success(f"✅ PINN training completed in {training_time:.1f} seconds!")
            except Exception as e:
                logger_phase.error(f"PINN training error: {str(e)}")
                st.error(f"PINN training failed: {str(e)}")
                return
    if 'pinn_model_phase' in st.session_state:
        st.subheader("📊 PINN Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Training Time", f"{st.session_state.pinn_training_time_phase:.1f}s")
        col2.metric("Final PDE Loss", f"{st.session_state.pinn_loss_history_phase['pde'][-1]:.2e}")
        col3.metric("Final IC Loss", f"{st.session_state.pinn_loss_history_phase['ic'][-1]:.2e}")
        loss_plot_path = plot_pinn_losses_phase(st.session_state.pinn_loss_history_phase, OUTPUT_DIR_PHASE)
        st.image(loss_plot_path, caption="Training Losses")
        with st.spinner("Evaluating PINN solution..."):
            times_to_evaluate = np.linspace(0, current_params['T_max'], 10)
            pinn_solution = evaluate_pinn_solution_phase(
                st.session_state.pinn_model_phase,
                times_to_evaluate,
                nx=current_params['nx'],
                ny=current_params['ny']
            )
            st.session_state.pinn_solution_phase = pinn_solution
        final_idx = -1
        conc_plot = plot_pinn_concentration_phase(pinn_solution, final_idx, OUTPUT_DIR_PHASE)
        energy_plot = plot_pinn_free_energy_phase(pinn_solution, final_idx, OUTPUT_DIR_PHASE)
        hist_plot = plot_pinn_histogram_phase(pinn_solution, final_idx, OUTPUT_DIR_PHASE)
        col1, col2 = st.columns(2)
        with col1:
            st.image(conc_plot, caption="Concentration Field")
        with col2:
            st.image(energy_plot, caption="Free Energy Density")
        st.image(hist_plot, caption="Concentration Distribution")
        st.subheader("🎬 Time Evolution")
        time_slider = st.slider(
            "Select time step",
            0, len(times_to_evaluate)-1,
            len(times_to_evaluate)-1,
            format="t = %.1f"
        )
        if time_slider != final_idx:
            conc_plot_ts = plot_pinn_concentration_phase(pinn_solution, time_slider, OUTPUT_DIR_PHASE)
            energy_plot_ts = plot_pinn_free_energy_phase(pinn_solution, time_slider, OUTPUT_DIR_PHASE)
            hist_plot_ts = plot_pinn_histogram_phase(pinn_solution, time_slider, OUTPUT_DIR_PHASE)
            col1, col2 = st.columns(2)
            with col1:
                st.image(conc_plot_ts, caption=f"Concentration (t = {times_to_evaluate[time_slider]:.1f})")
            with col2:
                st.image(energy_plot_ts, caption=f"Free Energy (t = {times_to_evaluate[time_slider]:.1f})")
            st.image(hist_plot_ts, caption=f"Concentration Distribution (t = {times_to_evaluate[time_slider]:.1f})")
        st.subheader("💾 Download PINN Results")
        model_path = os.path.join(OUTPUT_DIR_PHASE, 'pinn_model.pth')
        torch.save(st.session_state.pinn_model_phase.state_dict(), model_path)
        with open(model_path, 'rb') as f:
            st.download_button(
                "📥 Download Trained Model",
                f.read(),
                "pinn_model.pth",
                "application/octet-stream"
            )
        solution_path = os.path.join(OUTPUT_DIR_PHASE, 'pinn_solution.pkl')
        with open(solution_path, 'wb') as f:
            pickle.dump(pinn_solution, f)
        with open(solution_path, 'rb') as f:
            st.download_button(
                "📥 Download Solution Data",
                f.read(),
                "pinn_solution.pkl",
                "application/octet-stream"
            )
        loss_history_path = os.path.join(OUTPUT_DIR_PHASE, 'pinn_loss_history.pkl')
        with open(loss_history_path, 'wb') as f:
            pickle.dump(st.session_state.pinn_loss_history_phase, f)
        with open(loss_history_path, 'rb') as f:
            st.download_button(
                "📥 Download Loss History",
                f.read(),
                "pinn_loss_history.pkl",
                "application/octet-stream"
            )


# =====================================================
# Original Streamlit App for Phase Field (with PINN integration)
# =====================================================
def main_phase():
    st.set_page_config(page_title="Phase Field Simulation", page_icon="🔬", layout="wide")
    st.title("🔬 Phase Field Simulation: Spinodal Decomposition")
    st.markdown("""
    This interactive simulation demonstrates **phase decomposition** using the Cahn-Hilliard equation.
    Adjust parameters to see how they affect phase separation dynamics.
    """)
    if 'sim_phase' not in st.session_state:
        st.session_state.sim_phase = PhaseFieldSimulation(nx=256, ny=256, dx=1.0, dt=0.1)
        st.session_state.sim_phase.initialize_random_phase(c0=0.5, noise_amplitude=0.05)
    sim = st.session_state.sim_phase
    with st.sidebar:
        st.header("🎛️ Simulation Controls")
        st.subheader("Simulation Parameters")
        steps_to_run = st.number_input("Steps per update", min_value=1, max_value=1000, value=10)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Run Steps", use_container_width=True):
                with st.spinner("Running simulation..."):
                    sim.run_steps_phase(steps_to_run)
        with col2:
            if st.button("⏹️ Stop", use_container_width=True):
                st.rerun()
        if st.button("🔄 Reset Random", use_container_width=True):
            sim.initialize_random_phase(c0=0.5, noise_amplitude=0.05)
            st.rerun()
        if st.button("🌱 Reset Seed", use_container_width=True):
            sim.initialize_seed_phase(c0=0.3, seed_value=0.7, radius=15)
            st.rerun()
        st.divider()
        st.subheader("Free Energy Parameters")
        use_standard_double_well = st.checkbox("Use standard double-well (f(c)=W*c²(1-c)²)", value=True)
        if use_standard_double_well:
            W = st.slider("Double-well height (W)", 0.1, 5.0, 1.0, 0.1,
                          help="Controls barrier between phases. Higher = sharper interfaces.")
            sim.set_parameters_phase(W=W, A=None, B=None, C=None)
        else:
            colA, colB, colC = st.columns(3)
            with colA:
                A = st.slider("A coefficient", 0.1, 5.0, 1.0, 0.1)
            with colB:
                B = st.slider("B coefficient", -5.0, 0.0, -2.0, 0.1)
            with colC:
                C = st.slider("C coefficient", 0.1, 5.0, 1.0, 0.1)
            sim.set_parameters_phase(A=A, B=B, C=C)
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
            sim.initialize_random_phase(c0=c0, noise_amplitude=noise)
            st.rerun()
        st.divider()
    stats = sim.get_statistics_phase()
    st.subheader("📊 Current Statistics")
    st.metric("Time", f"{stats['time']:.1f}")
    st.metric("Step", f"{stats['step']}")
    st.metric("Mean Concentration", f"{stats['mean_concentration']:.4f}")
    st.metric("Std Dev", f"{stats['std_concentration']:.4f}")
    st.metric("High Phase Fraction", f"{stats['phase_fraction_high']:.3f}")
    st.metric("Low Phase Fraction", f"{stats['phase_fraction_low']:.3f}")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Concentration Field")
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
        energy = sim.compute_free_energy_density_phase()
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        im2 = ax2.imshow(energy, cmap='viridis', origin='lower')
        ax2.set_title("Free Energy Density")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        plt.colorbar(im2, ax=ax2, label="Energy Density")
        st.pyplot(fig2)
        plt.close(fig2)
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
    st.subheader("📈 Time Evolution")
    if len(sim.history['time']) > 1:
        fig4, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(sim.history['time'], sim.history['mean'], 'b-', linewidth=2)
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("Mean Concentration")
        axes[0].set_title("Mean Concentration vs Time")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(sim.history['time'], sim.history['std'], 'r-', linewidth=2)
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Standard Deviation")
        axes[1].set_title("Standard Deviation vs Time")
        axes[1].grid(True, alpha=0.3)
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
            csv_data = "time,mean_concentration,std_concentration,phase_fraction_high,phase_fraction_low\n"
            for i in range(len(sim.history['time'])):
                csv_data += f"{sim.history['time'][i]},{sim.history['mean'][i]},{sim.history['std'][i]},{sim.history['phase_high'][i]},{sim.history['phase_low'][i]}\n"
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="phase_field_statistics.csv",
                mime="text/csv"
            )
    with st.expander("ℹ️ About this Simulation"):
        st.markdown("""
        ## Phase Field Method for Spinodal Decomposition
        This simulation implements the **Cahn-Hilliard equation** to model phase separation in binary systems:
        ∂c/∂t = ∇·[M ∇μ]
        μ = df/dc - κ∇²c
        f(c) = A·c² + B·c³ + C·c⁴

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
                    sim.run_step_phase()
                    progress_bar.progress((i + 1) / auto_steps)
                    status_text.text(f"Running step {sim.step}...")
                    if (i + 1) % 5 == 0:
                        st.rerun()
    st.divider()
    add_pinn_section_phase()


# =====================================================
# Top-level Main to Select Simulation
# =====================================================
def main():
    st.title("Combined PINN Simulations")
    simulation_type = st.selectbox("Choose Simulation", ["Cu-Ni Diffusion", "Phase Field Spinodal Decomposition"])
    if simulation_type == "Cu-Ni Diffusion":
        main_diffusion()
    else:
        main_phase()


if __name__ == "__main__":
    main()
