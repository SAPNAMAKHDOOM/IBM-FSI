#!/usr/bin/env python3
"""
ibm_filament_animation.py

IBM-FSI simulation with:
- Step-by-step snapshot saving
- Final composite figure
- Optional GIF animation

Author: Dr. Sapna Makhdoom
Date: September 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time
from scipy.fft import fft2, ifft2, fftfreq
import warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path

# Try to import imageio for GIF creation
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("ImageIO not installed. GIF creation will be skipped.")
    print("Install with: pip install imageio")

# ----------------------------------------------------------------------
# 1. Parameters
# ----------------------------------------------------------------------

# Fluid domain
nx, ny = 64, 64
Lx, Ly = 4.0, 2.0
Re = 80.0

# Filament properties
N_points = 48
L_filament = 1.2
EI = 0.002
T0 = 0.03

# Flow forcing
U_inf = 0.8

# Time stepping
dt = 0.0002
nt = 5000
save_every = 50          # Save snapshot every 50 steps (100 snapshots total)
plot_every = 500
snapshot_every = 100

# Damping
damping = 0.3

# ----------------------------------------------------------------------
# 2. Grid setup
# ----------------------------------------------------------------------

dx = Lx / nx
dy = Ly / ny

x = np.linspace(0, Lx, nx, endpoint=False) + dx/2
y = np.linspace(0, Ly, ny, endpoint=False) + dy/2
X, Y = np.meshgrid(x, y, indexing='ij')

kx = 2.0 * np.pi * fftfreq(nx, Lx/nx)
ky = 2.0 * np.pi * fftfreq(ny, Ly/ny)
Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
K2 = Kx**2 + Ky**2
K2[0, 0] = 1.0

# Create directories for outputs
snapshot_dir = Path("snapshots")
snapshot_dir.mkdir(exist_ok=True)

composite_dir = Path("composite")
composite_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# 3. Initial filament position
# ----------------------------------------------------------------------

s = np.linspace(0, L_filament, N_points)
X_fil = 0.6 * np.ones(N_points)
Y_fil = 0.5 + s / L_filament * 1.0

u_fil = np.zeros(N_points)
v_fil = 0.03 * np.sin(3 * np.pi * s / L_filament)

# ----------------------------------------------------------------------
# 4. Fluid fields
# ----------------------------------------------------------------------

u = U_inf * np.ones((nx, ny))
v = np.zeros((nx, ny))

# ----------------------------------------------------------------------
# 5. Helper functions
# ----------------------------------------------------------------------

def delta_function(r):
    r = np.abs(r)
    if r < 1.0:
        return (3.0 - 2.0*r + np.sqrt(1.0 + 4.0*r - 4.0*r**2)) / 8.0
    elif r < 2.0:
        return (5.0 - 2.0*r - np.sqrt(-7.0 + 12.0*r - 4.0*r**2)) / 8.0
    else:
        return 0.0

def spread_force_to_grid(force_x, force_y, X_fil, Y_fil, dx, dy):
    Fx = np.zeros((nx, ny))
    Fy = np.zeros((nx, ny))
    
    for k in range(N_points):
        if np.isnan(X_fil[k]) or np.isnan(Y_fil[k]):
            continue
            
        x_idx = X_fil[k] / dx
        y_idx = Y_fil[k] / dy
        
        x_idx = np.clip(x_idx, 0, nx-1)
        y_idx = np.clip(y_idx, 0, ny-1)
        
        i_start = int(np.floor(x_idx)) - 1
        i_end = int(np.floor(x_idx)) + 2
        j_start = int(np.floor(y_idx)) - 1
        j_end = int(np.floor(y_idx)) + 2
        
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                i_mod = i % nx
                j_mod = j % ny
                
                dx_grid = (i - x_idx)
                dy_grid = (j - y_idx)
                
                d_x = delta_function(dx_grid)
                d_y = delta_function(dy_grid)
                d_val = d_x * d_y / (dx * dy)
                
                Fx[i_mod, j_mod] += force_x[k] * d_val
                Fy[i_mod, j_mod] += force_y[k] * d_val
    
    return Fx, Fy

def interpolate_velocity_to_filament(X_fil, Y_fil, u, v, dx, dy):
    u_fil = np.zeros(N_points)
    v_fil = np.zeros(N_points)
    
    for k in range(N_points):
        if np.isnan(X_fil[k]) or np.isnan(Y_fil[k]):
            continue
            
        x_idx = X_fil[k] / dx
        y_idx = Y_fil[k] / dy
        
        x_idx = np.clip(x_idx, 0, nx-1)
        y_idx = np.clip(y_idx, 0, ny-1)
        
        i_start = int(np.floor(x_idx)) - 1
        i_end = int(np.floor(x_idx)) + 2
        j_start = int(np.floor(y_idx)) - 1
        j_end = int(np.floor(y_idx)) + 2
        
        u_sum = 0.0
        v_sum = 0.0
        
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                i_mod = i % nx
                j_mod = j % ny
                
                dx_grid = (i - x_idx)
                dy_grid = (j - y_idx)
                
                d_x = delta_function(dx_grid)
                d_y = delta_function(dy_grid)
                d_val = d_x * d_y
                
                u_sum += u[i_mod, j_mod] * d_val
                v_sum += v[i_mod, j_mod] * d_val
        
        u_fil[k] = u_sum
        v_fil[k] = v_sum
    
    return u_fil, v_fil

def compute_filament_forces(X_fil, Y_fil, EI, T0, s, L_filament):
    ds = s[1] - s[0]
    N = len(X_fil)
    
    if np.any(np.isnan(X_fil)) or np.any(np.isnan(Y_fil)):
        return np.zeros(N), np.zeros(N)
    
    dX_ds = np.gradient(X_fil, ds, edge_order=2)
    dY_ds = np.gradient(Y_fil, ds, edge_order=2)
    
    d2X_ds4 = np.gradient(np.gradient(np.gradient(np.gradient(X_fil, ds), ds), ds), ds)
    d2Y_ds4 = np.gradient(np.gradient(np.gradient(np.gradient(Y_fil, ds), ds), ds), ds)
    
    Fx_bend = -EI * d2X_ds4
    Fy_bend = -EI * d2Y_ds4
    
    stretch = np.sqrt(dX_ds**2 + dY_ds**2) - 1.0
    Fx_tension = T0 * np.gradient(stretch * dX_ds, ds)
    Fy_tension = T0 * np.gradient(stretch * dY_ds, ds)
    
    Fx = Fx_bend + Fx_tension
    Fy = Fy_bend + Fy_tension
    
    Fx[:2] = 0.0
    Fy[:2] = 0.0
    
    Fx = np.clip(Fx, -2.0, 2.0)
    Fy = np.clip(Fy, -2.0, 2.0)
    
    return Fx, Fy

def solve_fluid(u, v, Fx, Fy, dt, Re, U_inf):
    u_hat = fft2(u)
    v_hat = fft2(v)
    Fx_hat = fft2(Fx)
    Fy_hat = fft2(Fy)
    
    du_dx = ifft2c(1j * Kx * u_hat)
    du_dy = ifft2c(1j * Ky * u_hat)
    dv_dx = ifft2c(1j * Kx * v_hat)
    dv_dy = ifft2c(1j * Ky * v_hat)
    
    adv_x = u * du_dx + v * du_dy
    adv_y = u * dv_dx + v * dv_dy
    
    adv_x_hat = fft2(adv_x)
    adv_y_hat = fft2(adv_y)
    
    visc_x_hat = (1.0 / Re) * (-K2) * u_hat
    visc_y_hat = (1.0 / Re) * (-K2) * v_hat
    
    body_force_x_hat = Fx_hat
    body_force_y_hat = Fy_hat
    
    body_force_x_hat[0, 0] += U_inf / dt
    
    du_hat_dt = -adv_x_hat + visc_x_hat + body_force_x_hat
    dv_hat_dt = -adv_y_hat + visc_y_hat + body_force_y_hat
    
    div_u_hat = 1j * Kx * du_hat_dt + 1j * Ky * dv_hat_dt
    du_hat_dt -= (Kx / K2) * div_u_hat
    dv_hat_dt -= (Ky / K2) * div_u_hat
    
    u_hat_new = u_hat + dt * du_hat_dt
    v_hat_new = v_hat + dt * dv_hat_dt
    
    u_hat_new[0, 0] = 0.0
    v_hat_new[0, 0] = 0.0
    
    u_new = ifft2c(u_hat_new)
    v_new = ifft2c(v_hat_new)
    
    return u_new, v_new

def ifft2c(f_hat):
    return np.fft.ifft2(f_hat).real

def save_snapshot(step, t, u, v, X_fil, Y_fil):
    """Save a single snapshot as a PNG file."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    omega = ifft2c(1j * Kx * fft2(v) - 1j * Ky * fft2(u))
    ax.contourf(X, Y, omega, levels=30, cmap=cm.RdBu_r, alpha=0.6)
    ax.plot(X_fil, Y_fil, 'k-', linewidth=3)
    ax.set_xlim([0, Lx])
    ax.set_ylim([0, Ly])
    ax.set_aspect('equal')
    ax.set_title(f'Flexible Filament, t={t:.3f}', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Add time stamp
    ax.text(0.05, 0.95, f'Time: {t:.3f}s', transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filename = snapshot_dir / f"snapshot_{step:06d}.png"
    plt.savefig(filename, dpi=100)
    plt.close(fig)
    return filename

def create_composite_figure(snapshots, times, save_path):
    """Create a composite figure with all snapshots in a grid."""
    n_snapshots = len(snapshots)
    if n_snapshots == 0:
        return
    
    # Determine grid size (e.g., 3 columns)
    cols = 4
    rows = (n_snapshots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, ax in enumerate(axes):
        if idx < n_snapshots:
            # Load and display the snapshot image
            img = plt.imread(snapshots[idx])
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f't={times[idx]:.3f}s', fontsize=10)
        else:
            ax.axis('off')
    
    plt.suptitle('Flexible Filament Evolution Over Time', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Composite figure saved: {save_path}")

def create_gif(snapshot_files, output_path, fps=10):
    """Create an animated GIF from snapshot images."""
    if not HAS_IMAGEIO:
        print("ImageIO not available. Skipping GIF creation.")
        return
    
    images = []
    for file in snapshot_files:
        if file.exists():
            images.append(imageio.imread(file))
    
    if images:
        imageio.mimsave(output_path, images, fps=fps)
        print(f"GIF saved: {output_path}")

# ----------------------------------------------------------------------
# 6. Main simulation
# ----------------------------------------------------------------------

def main():
    global u, v, X_fil, Y_fil, u_fil, v_fil
    
    print("=" * 60)
    print("IBM-FSI: Flexible Filament with Animation")
    print("=" * 60)
    print(f"Grid: {nx}x{ny}, Re={Re}, dt={dt}, nt={nt}")
    print(f"Filament: {N_points} points, L={L_filament}, EI={EI}")
    print(f"U_inf={U_inf}, damping={damping}")
    print(f"Saving snapshots every {save_every} steps")
    print("=" * 60)

    times = []
    KE = []
    snapshot_files = []
    snapshot_times = []
    
    start_time = time.time()
    
    for step in range(nt + 1):
        # Compute forces
        Fx_fil, Fy_fil = compute_filament_forces(X_fil, Y_fil, EI, T0, s, L_filament)
        
        # Spread forces
        Fx_grid, Fy_grid = spread_force_to_grid(Fx_fil, Fy_fil, X_fil, Y_fil, dx, dy)
        
        # Solve fluid
        u, v = solve_fluid(u, v, Fx_grid, Fy_grid, dt, Re, U_inf)
        
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            print(f"ERROR: Fluid became NaN at step {step}")
            break
        
        # Interpolate velocity
        u_fil_interp, v_fil_interp = interpolate_velocity_to_filament(X_fil, Y_fil, u, v, dx, dy)
        
        if np.any(np.isnan(u_fil_interp)) or np.any(np.isnan(v_fil_interp)):
            print(f"ERROR: Interpolation returned NaN at step {step}")
            break
        
        # Update filament
        u_fil = u_fil_interp * (1 - damping) + u_fil * damping
        v_fil = v_fil_interp * (1 - damping) + v_fil * damping
        
        X_fil += dt * u_fil
        Y_fil += dt * v_fil
        
        X_fil = np.clip(X_fil, 0.1, Lx - 0.1)
        Y_fil = np.clip(Y_fil, 0.1, Ly - 0.1)
        
        # Clamped at bottom
        X_fil[0] = 0.6
        Y_fil[0] = 0.5
        X_fil[1] = X_fil[0] + 0.01
        Y_fil[1] = Y_fil[0] + 0.01
        
        # Store data
        if step % save_every == 0:
            times.append(step * dt)
            KE.append(0.5 * np.mean(u**2 + v**2))
        
        # SAVE SNAPSHOT
        if step % snapshot_every == 0:
            t = step * dt
            filename = save_snapshot(step, t, u, v, X_fil, Y_fil)
            snapshot_files.append(filename)
            snapshot_times.append(t)
            print(f"Snapshot {len(snapshot_files)}: t={t:.3f}s")
        
        # Progress
        if step % plot_every == 0 and step > 0:
            print(f"Step {step}/{nt}, t={step*dt:.3f}, KE={KE[-1]:.6f}")
    
    end_time = time.time()
    print(f"\nSimulation finished in {end_time - start_time:.2f} seconds.")
    print(f"Saved {len(snapshot_files)} snapshots.")
    
    # ------------------------------------------------------------------
    # Create composite figure
    # ------------------------------------------------------------------
    print("\nCreating composite figure...")
    composite_path = composite_dir / "composite_all_snapshots.png"
    create_composite_figure(snapshot_files, snapshot_times, composite_path)
    
    # ------------------------------------------------------------------
    # Create GIF animation
    # ------------------------------------------------------------------
    print("\nCreating GIF animation...")
    gif_path = composite_dir / "filament_animation.gif"
    create_gif(snapshot_files, gif_path, fps=5)
    
    # ------------------------------------------------------------------
    # Energy plot
    # ------------------------------------------------------------------
    print("\nCreating energy plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(times, KE, 'b-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Kinetic Energy', fontsize=12)
    ax.grid(True)
    ax.set_title('Kinetic Energy Evolution', fontsize=14)
    plt.tight_layout()
    plt.savefig(composite_dir / "energy_evolution.png", dpi=150)
    plt.close()
    
    print(f"\nFinal KE = {KE[-1]:.6f}")
    print("=" * 60)
    print("All outputs saved to:")
    print(f"  - Snapshots: {snapshot_dir}/")
    print(f"  - Composite: {composite_path}")
    print(f"  - GIF: {gif_path}")
    print("=" * 60)
    print("Simulation completed successfully!")

if __name__ == "__main__":
    main()