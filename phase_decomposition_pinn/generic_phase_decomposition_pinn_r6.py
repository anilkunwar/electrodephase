import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os
import time
import pickle
import io
import zipfile
import logging
from matplotlib import cm

# ==============================
# GLOBAL CONFIGURATION
# ==============================
# Set device and seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = "/tmp/pinn_ch_solution"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "training.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Physical parameters (match your phase-field app)
Lx = Ly = 64.0          # Domain size
T_max = 10.0            # Max simulation time (keep short for PINN stability)
W = 1.0
A = W
B = -2.0 * W
C = W
kappa = 2.0
M = 1.0

# Numerical settings
nx_eval = ny_eval = 64  # Evaluation grid
N_ic = 2000             # Initial condition points
N_pde = 5000            # PDE collocation points
epochs = 5000
lr = 1e-3
ic_weight = 100.0
phys_weight = 1.0

# ==============================
# NEURAL NETWORK
# ==============================
class CahnHilliardPINN(nn.Module):
    def __init__(self, layers=[5, 128, 128, 128, 1], activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'activation_{i}', activation())
        self.net.add_module('output', nn.Linear(layers[-2], layers[-1]))
        self.sigmoid = nn.Sigmoid()  # Enforce c ∈ (0,1)

    def forward(self, x, y, t):
        # Periodic embedding: sin/cos for x and y
        x_sin = torch.sin(2 * np.pi * x / Lx)
        x_cos = torch.cos(2 * np.pi * x / Lx)
        y_sin = torch.sin(2 * np.pi * y / Ly)
        y_cos = torch.cos(2 * np.pi * y / Ly)
        t_norm = t / T_max  # Normalize time

        inp = torch.cat([x_sin, x_cos, y_sin, y_cos, t_norm], dim=1)
        out = self.net(inp)
        return self.sigmoid(out)

# ==============================
# PHYSICS RESIDUAL
# ==============================
def chemical_potential(c, A, B, C):
    return 2 * A * c + 3 * B * c**2 + 4 * C * c**3

def cahn_hilliard_residual(model, x, y, t, A, B, C, kappa, M):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    c = model(x, y, t)
    c = c.reshape(-1, 1)

    # ∂c/∂t
    c_t = torch.autograd.grad(c, t, grad_outputs=torch.ones_like(c), create_graph=True)[0]

    # ∇c
    c_x = torch.autograd.grad(c, x, grad_outputs=torch.ones_like(c), create_graph=True)[0]
    c_y = torch.autograd.grad(c, y, grad_outputs=torch.ones_like(c), create_graph=True)[0]

    # ∇²c
    c_xx = torch.autograd.grad(c_x, x, grad_outputs=torch.ones_like(c_x), create_graph=True)[0]
    c_yy = torch.autograd.grad(c_y, y, grad_outputs=torch.ones_like(c_y), create_graph=True)[0]
    lap_c = c_xx + c_yy

    # Chemical potential: μ = f'(c) - κ ∇²c
    mu_bulk = chemical_potential(c, A, B, C)
    mu = mu_bulk - kappa * lap_c

    # ∇μ
    mu_x = torch.autograd.grad(mu, x, grad_outputs=torch.ones_like(mu), create_graph=True)[0]
    mu_y = torch.autograd.grad(mu, y, grad_outputs=torch.ones_like(mu), create_graph=True)[0]

    # Flux = M ∇μ
    flux_x = M * mu_x
    flux_y = M * mu_y

    # ∇·(M ∇μ)
    flux_x_x = torch.autograd.grad(flux_x, x, grad_outputs=torch.ones_like(flux_x), create_graph=True)[0]
    flux_y_y = torch.autograd.grad(flux_y, y, grad_outputs=torch.ones_like(flux_y), create_graph=True)[0]
    div_flux = flux_x_x + flux_y_y

    residual = c_t - div_flux
    return residual

# ==============================
# LOSS FUNCTIONS
# ==============================
def initial_condition_loss(model, x, y, c0_field):
    t0 = torch.zeros_like(x)
    c_pred = model(x, y, t0)
    return torch.mean((c_pred - c0_field) ** 2)

def physics_loss(model, x, y, t):
    residual = cahn_hilliard_residual(model, x, y, t, A, B, C, kappa, M)
    return torch.mean(residual ** 2)

# ==============================
# TRAINING FUNCTION
# ==============================
def train_pinn():
    logger.info("Initializing PINN training...")
    model = CahnHilliardPINN().to(device)

    # Initial condition: c0 = 0.5 + noise
    c0_val = 0.5
    noise_amp = 0.05
    x_ic = torch.rand(N_ic, 1) * Lx
    y_ic = torch.rand(N_ic, 1) * Ly
    c0_ic = c0_val + noise_amp * (2 * torch.rand(N_ic, 1) - 1)
    x_ic, y_ic, c0_ic = x_ic.to(device), y_ic.to(device), c0_ic.to(device)

    # PDE collocation points
    x_pde = torch.rand(N_pde, 1) * Lx
    y_pde = torch.rand(N_pde, 1) * Ly
    t_pde = torch.rand(N_pde, 1) * T_max
    x_pde, y_pde, t_pde = x_pde.to(device), y_pde.to(device), t_pde.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

    loss_history = {
        'epoch': [],
        'total': [],
        'physics': [],
        'initial': []
    }

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_ic = initial_condition_loss(model, x_ic, y_ic, c0_ic)
        loss_phys = physics_loss(model, x_pde, y_pde, t_pde)

        loss = ic_weight * loss_ic + phys_weight * loss_phys

        try:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch}: {e}")
            st.error(f"Training failed: {e}")
            return None, None

        if (epoch + 1) % 100 == 0:
            loss_history['epoch'].append(epoch + 1)
            loss_history['total'].append(loss.item())
            loss_history['physics'].append(loss_phys.item())
            loss_history['initial'].append(loss_ic.item())

            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(
                f"Epoch {epoch+1}/{epochs} | "
                f"Total: {loss.item():.6f} | "
                f"Physics: {loss_phys.item():.6f} | "
                f"IC: {loss_ic.item():.6f}"
            )

        scheduler.step()

    progress_bar.progress(1.0)
    status_text.text("✅ Training completed!")
    logger.info("Training finished successfully.")
    return model, loss_history

# ==============================
# EVALUATION & VALIDATION
# ==============================
def evaluate_model(model, time_val):
    x = torch.linspace(0, Lx, nx_eval, device=device)
    y = torch.linspace(0, Ly, ny_eval, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    X_flat = X.flatten().unsqueeze(1)
    Y_flat = Y.flatten().unsqueeze(1)
    T_flat = torch.full_like(X_flat, time_val)

    with torch.no_grad():
        c_pred = model(X_flat, Y_flat, T_flat).cpu().numpy().reshape(nx_eval, ny_eval)
    return X.cpu().numpy(), Y.cpu().numpy(), c_pred

def validate_initial_condition(model, tolerance=1e-2):
    x_test = torch.rand(500, 1) * Lx
    y_test = torch.rand(500, 1) * Ly
    c0_true = 0.5 + 0.05 * (2 * torch.rand(500, 1) - 1)
    c0_pred = model(x_test.to(device), y_test.to(device), torch.zeros_like(x_test).to(device))
    error = torch.mean((c0_pred.cpu() - c0_true) ** 2).item()
    return error < tolerance, error

# ==============================
# PLOTTING UTILITIES
# ==============================
def plot_loss_history(loss_hist):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(loss_hist['epoch'], loss_hist['total'], label='Total Loss', linewidth=2)
    ax.plot(loss_hist['epoch'], loss_hist['physics'], label='Physics Loss', linewidth=1.5, linestyle='--')
    ax.plot(loss_hist['epoch'], loss_hist['initial'], label='Initial Condition Loss', linewidth=1.5, linestyle='-.')
    ax.set_yscale('log')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Loss History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_concentration_fields(model, times):
    n = len(times)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for idx, t in enumerate(times):
        X, Y, c = evaluate_model(model, t)
        # Top row: concentration
        im1 = axes[0, idx].imshow(c.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='bwr', vmin=0, vmax=1)
        axes[0, idx].set_title(f'Concentration at t={t:.1f}')
        plt.colorbar(im1, ax=axes[0, idx], fraction=0.046, pad=0.04)

        # Bottom row: free energy density
        f = A * c**2 + B * c**3 + C * c**4
        im2 = axes[1, idx].imshow(f.T, origin='lower', extent=[0, Lx, 0, Ly], cmap='viridis')
        axes[1, idx].set_title(f'Free Energy at t={t:.1f}')
        plt.colorbar(im2, ax=axes[1, idx], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

def plot_histogram(model, t=0.0):
    _, _, c = evaluate_model(model, t)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(c.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Concentration c')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Concentration Distribution at t={t:.1f}')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    return fig

# ==============================
# EXPORT UTILITIES
# ==============================
def save_solution(model, loss_hist, times):
    solution = {
        'params': {
            'Lx': Lx, 'Ly': Ly, 'T_max': T_max,
            'A': A, 'B': B, 'C': C, 'kappa': kappa, 'M': M,
            'nx': nx_eval, 'ny': ny_eval
        },
        'loss_history': loss_hist,
        'times': times,
        'model_state_dict': model.state_dict()
    }

    # Save full solution
    sol_path = os.path.join(OUTPUT_DIR, "cahn_hilliard_pinn_solution.pkl")
    with open(sol_path, 'wb') as f:
        pickle.dump(solution, f)
    logger.info(f"Solution saved to {sol_path}")

    # Save TorchScript model
    scripted = torch.jit.script(model.cpu())
    scripted.save(os.path.join(OUTPUT_DIR, "cahn_hilliard_pinn_model.pt"))

    return sol_path

def create_zip_download():
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, fname)
            zf.write(fpath, fname)
    return zip_buffer.getvalue()

# ==============================
# STREAMLIT APP
# ==============================
def main():
    st.set_page_config(page_title="PINN: Cahn–Hilliard", layout="wide")
    st.title("🧫 Physics-Informed Neural Network for Cahn–Hilliard Equation")
    st.markdown("""
    This app solves the **Cahn–Hilliard equation** using a **Physics-Informed Neural Network (PINN)**.
    It models **spinodal decomposition** in a binary mixture with periodic boundaries.
    """)

    # Initialize session state
    if 'trained' not in st.session_state:
        st.session_state.trained = False
        st.session_state.model = None
        st.session_state.loss_hist = None

    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Parameters")
        st.markdown(f"**Domain**: {Lx} × {Ly}")
        st.markdown(f"**Time**: [0, {T_max}]")
        st.markdown(f"**Epochs**: {epochs}")
        st.markdown(f"**Device**: {device}")

        if st.button("▶️ Train PINN", type="primary"):
            with st.spinner("Training in progress..."):
                start = time.time()
                model, loss_hist = train_pinn()
                if model is not None:
                    st.session_state.trained = True
                    st.session_state.model = model
                    st.session_state.loss_hist = loss_hist
                    st.session_state.train_time = time.time() - start
                    st.success(f"✅ Training completed in {st.session_state.train_time:.1f}s!")
                else:
                    st.error("Training failed!")

        if st.session_state.trained:
            if st.button("💾 Export All Results"):
                sol_path = save_solution(
                    st.session_state.model,
                    st.session_state.loss_hist,
                    [0.0, 2.0, 5.0, T_max]
                )
                zip_data = create_zip_download()
                st.download_button(
                    "📥 Download ZIP",
                    zip_data,
                    file_name="pinn_ch_solution.zip",
                    mime="application/zip"
                )

    # Main content
    if st.session_state.trained:
        model = st.session_state.model
        loss_hist = st.session_state.loss_hist

        # 1. Loss History
        st.subheader("📉 Training Loss")
        fig_loss = plot_loss_history(loss_hist)
        st.pyplot(fig_loss)

        # 2. Validation
        st.subheader("🔍 Validation")
        valid, error = validate_initial_condition(model)
        st.metric("Initial Condition Error", f"{error:.4f}", delta=None, delta_color="inverse")
        st.write("✅ Initial condition satisfied" if valid else "⚠️ Initial condition error above tolerance")

        # 3. Solution Visualization
        st.subheader("🖼️ Solution Visualization")
        times = st.multiselect(
            "Select time steps to visualize:",
            options=[0.0, 2.0, 5.0, T_max],
            default=[0.0, T_max]
        )
        if times:
            fig_sol = plot_concentration_fields(model, times)
            st.pyplot(fig_sol)

        # 4. Histogram
        t_hist = st.slider("Time for histogram", 0.0, float(T_max), 0.0, 0.5)
        fig_hist = plot_histogram(model, t_hist)
        st.pyplot(fig_hist)

        # 5. Training Log
        with st.expander("📋 Training Log"):
            log_path = os.path.join(OUTPUT_DIR, "training.log")
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    st.text(f.read()[-2000:])  # last 2000 chars

    else:
        st.info("Click **▶️ Train PINN** in the sidebar to start the simulation.")

    # Footer
    st.markdown("---")
    st.markdown("""
    **Note**: PINNs for fourth-order PDEs like Cahn–Hilliard are **research-grade** and **less efficient**
    than finite-difference methods for forward simulation. This implementation is intended for
    **educational purposes**, **inverse problems**, or **irregular domains**.
    """)

if __name__ == "__main__":
    main()
