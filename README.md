# IBM-FSI
# IBM-FSI: Immersed Boundary Method for Fluid-Structure Interaction

A 2D Immersed Boundary Method (IBM) solver for simulating a flexible filament in uniform flow.

## 📖 Overview

This code simulates a flexible filament immersed in a uniform flow. The fluid is solved using a pseudo-spectral method on a periodic domain, while the filament is modeled as a 1D Euler-Bernoulli beam. The fluid and structure are coupled using Peskin's immersed boundary method with a discrete delta function.

## 🔬 Governing Physics

### Fluid (Navier-Stokes)
\[
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\nabla p + \frac{1}{Re} \nabla^2 \mathbf{u} + \mathbf{f}
\]
\[
\nabla \cdot \mathbf{u} = 0
\]

### Structure (Euler-Bernoulli Beam)
\[
\rho_s \frac{\partial^2 \mathbf{X}}{\partial t^2} = EI \frac{\partial^4 \mathbf{X}}{\partial s^4} - T \frac{\partial}{\partial s} \left( \frac{\partial \mathbf{X}}{\partial s} \right)
\]

### Coupling
- **Spreading**: Lagrangian forces are spread to the Eulerian grid using the discrete delta function.
- **Interpolation**: Fluid velocities are interpolated to the Lagrangian points.
- **No-slip condition**: The filament moves with the local fluid velocity.

## 🛠️ Numerical Methods

| Component | Method |
|-----------|--------|
| Fluid solver | Pseudo-spectral (FFT), periodic |
| Structure solver | Finite differences on Lagrangian grid |
| Coupling | Peskin's IBM with 4-point delta function |
| Time integration | Explicit Euler (small dt for stability) |
| Stability | Damping and clipping to prevent NaN |

## 🚀 How to Run

### Basic version (stable):
```bash
python ibm_filament_safe.py
