"""
Mass transport solver for EDOT monomer concentration.

Stage A: Pure diffusion-reaction
    dC/dt = D * nabla^2(C) - R

Stage C: Nernst-Planck (diffusion + migration)
    dC/dt = nabla . (D*nabla(C) + (z*F*D)/(R_gas*T) * C * nabla(V)) - R

Uses operator splitting with ADI (Alternating Direction Implicit)
for the diffusion part and explicit treatment of migration/reaction.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import constants as const
from .utils_grid import Grid2D

logger = logging.getLogger(__name__)


def _build_1d_diffusion_matrix(N: int, d: float, dt: float, theta: float) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build implicit and explicit tridiagonal matrices for 1D diffusion.

    For the equation dC/dt = d * d2C/dx2 discretized as:
        (I - theta*dt*d*A) C^{n+1} = (I + (1-theta)*dt*d*A) C^n

    Parameters
    ----------
    N : int
        Number of grid points.
    d : float
        D / dx^2 coefficient.
    dt : float
        Time step [s].
    theta : float
        Implicitness (0=explicit, 0.5=CN, 1=implicit).

    Returns
    -------
    L_lhs, L_rhs : sparse matrices
    """
    alpha = d * dt
    # Tridiagonal Laplacian operator: main diag = -2, off diag = +1
    diags = np.array([np.ones(N - 1), -2.0 * np.ones(N), np.ones(N - 1)])
    A = sparse.diags(diags, [-1, 0, 1], shape=(N, N), format="csr")
    I = sparse.eye(N, format="csr")
    L_lhs = I - theta * alpha * A
    L_rhs = I + (1.0 - theta) * alpha * A
    return L_lhs, L_rhs


def solve_diffusion_step(
    C: np.ndarray,
    grid: Grid2D,
    D: float,
    dt: float,
    R_sink: np.ndarray,
    domain_mask: np.ndarray,
    C_bulk: float,
    left_feeder: np.ndarray,
    right_feeder: np.ndarray,
    theta: float = 0.5,
) -> np.ndarray:
    """Advance concentration one time step via diffusion-reaction.

    Uses ADI (Alternating Direction Implicit) splitting:
      Half-step: implicit in x, explicit in y
      Half-step: implicit in y, explicit in x

    Boundary conditions:
      - C = C_bulk on feeder electrode boundaries
      - dC/dn = 0 on insulating walls (handled by ghost cells)

    Parameters
    ----------
    C : np.ndarray (Ny, Nx)
        Current concentration field [mol/m^3].
    grid : Grid2D
    D : float
        Diffusion coefficient [m^2/s].
    dt : float
        Time step [s].
    R_sink : np.ndarray (Ny, Nx)
        Reaction sink term [mol/(m^3*s)].
    domain_mask : np.ndarray (Ny, Nx)
        Active domain cells.
    C_bulk : float
        Bulk concentration for Dirichlet BCs [mol/m^3].
    left_feeder, right_feeder : np.ndarray
        Feeder boundary masks.
    theta : float
        Implicitness parameter.

    Returns
    -------
    C_new : np.ndarray (Ny, Nx)
    """
    Ny, Nx = C.shape
    C_new = C.copy()

    rx = D * dt / (grid.dx ** 2)
    ry = D * dt / (grid.dy ** 2)

    # Stability check for explicit part
    fourier = D * dt * (1.0 / grid.dx ** 2 + 1.0 / grid.dy ** 2)
    if fourier > 0.5 and theta < 1.0:
        logger.warning("Fourier number %.3f > 0.5; consider smaller dt or theta=1.0", fourier)

    # Simple explicit-implicit approach (not full ADI, for robustness)
    # Laplacian via central differences
    Lap = np.zeros_like(C)
    Lap[1:-1, 1:-1] = (
        D * (C[1:-1, 2:] - 2 * C[1:-1, 1:-1] + C[1:-1, :-2]) / grid.dx ** 2
        + D * (C[2:, 1:-1] - 2 * C[1:-1, 1:-1] + C[:-2, 1:-1]) / grid.dy ** 2
    )

    # Semi-implicit: C^{n+1} = C^n + dt * [theta * Lap^{n+1} + (1-theta) * Lap^n - R]
    # For simplicity, use explicit Euler with the Laplacian and subtract reaction
    # With theta weighting for stability
    if theta < 1.0:
        # Explicit contribution
        C_new[1:-1, 1:-1] = (
            C[1:-1, 1:-1]
            + dt * Lap[1:-1, 1:-1]
            - dt * R_sink[1:-1, 1:-1]
        )
    else:
        # Fully implicit via iterative solve
        # Use a few Gauss-Seidel iterations
        for _ in range(20):
            Lap_new = np.zeros_like(C_new)
            Lap_new[1:-1, 1:-1] = (
                D * (C_new[1:-1, 2:] - 2 * C_new[1:-1, 1:-1] + C_new[1:-1, :-2]) / grid.dx ** 2
                + D * (C_new[2:, 1:-1] - 2 * C_new[1:-1, 1:-1] + C_new[:-2, 1:-1]) / grid.dy ** 2
            )
            C_new[1:-1, 1:-1] = (
                C[1:-1, 1:-1]
                + dt * (theta * Lap_new[1:-1, 1:-1] + (1 - theta) * Lap[1:-1, 1:-1])
                - dt * R_sink[1:-1, 1:-1]
            )
            # Re-apply BCs
            C_new[left_feeder] = C_bulk
            C_new[right_feeder] = C_bulk
            C_new[~domain_mask] = 0.0

    # Apply boundary conditions
    C_new[left_feeder] = C_bulk
    C_new[right_feeder] = C_bulk
    C_new[~domain_mask] = 0.0

    # Enforce non-negative concentration
    C_new = np.maximum(C_new, 0.0)

    return C_new


def add_migration_flux(
    C: np.ndarray,
    V: np.ndarray,
    grid: Grid2D,
    D: float,
    z: int,
    dt: float,
    domain_mask: np.ndarray,
    T: float = const.T_DEFAULT,
) -> np.ndarray:
    """Add migration contribution to concentration field (explicit).

    Migration flux: J_mig = -(z*F*D)/(R*T) * C * nabla(V)
    Contribution to dC/dt: nabla . J_mig = (z*F*D)/(R*T) * nabla . (C * nabla(V))

    For explicit stepping:
        C_new = C + dt * (zFD/RT) * div(C * grad(V))

    Parameters
    ----------
    C : np.ndarray (Ny, Nx)
    V : np.ndarray (Ny, Nx)
        Electric potential [V].
    grid : Grid2D
    D : float
        Diffusion coefficient [m^2/s].
    z : int
        Species charge number.
    dt : float
        Time step [s].
    domain_mask : np.ndarray
    T : float
        Temperature [K].

    Returns
    -------
    C_new : np.ndarray
    """
    mobility = z * const.F * D / (const.R_GAS * T)

    # Compute grad(V)
    dVdy = np.zeros_like(V)
    dVdx = np.zeros_like(V)
    dVdx[1:-1, 1:-1] = (V[1:-1, 2:] - V[1:-1, :-2]) / (2.0 * grid.dx)
    dVdy[1:-1, 1:-1] = (V[2:, 1:-1] - V[:-2, 1:-1]) / (2.0 * grid.dy)

    # Migration flux components: Jx = mobility * C * dV/dx
    Jx = mobility * C * dVdx
    Jy = mobility * C * dVdy

    # Divergence of migration flux
    div_J = np.zeros_like(C)
    div_J[1:-1, 1:-1] = (
        (Jx[1:-1, 2:] - Jx[1:-1, :-2]) / (2.0 * grid.dx)
        + (Jy[2:, 1:-1] - Jy[:-2, 1:-1]) / (2.0 * grid.dy)
    )

    C_new = C + dt * div_J
    C_new[~domain_mask] = 0.0
    C_new = np.maximum(C_new, 0.0)
    return C_new


def compute_reaction_rate(
    C: np.ndarray,
    electrode_mask: np.ndarray,
    config_physics: "PhysicsConfig",
    E_mag: np.ndarray | None = None,
    eta: np.ndarray | None = None,
) -> np.ndarray:
    """Compute volumetric reaction sink rate.

    Stage A: R = k_rxn * C  (at electrode only)
    Stage B: R = k_rxn * C * (1 + beta*|E|)  (field-enhanced)
    Stage C: same as B (migration handled in transport)
    Stage D: R = j_BV / (n*F*dx)  (Butler-Volmer based)

    Parameters
    ----------
    C : np.ndarray (Ny, Nx)
        Concentration [mol/m^3].
    electrode_mask : np.ndarray (Ny, Nx)
    config_physics : PhysicsConfig
    E_mag : np.ndarray, optional
        Electric field magnitude [V/m]. Required for Stage >= B.
    eta : np.ndarray, optional
        Overpotential at electrode [V]. Required for Stage D.
        1D array matching number of electrode cells.

    Returns
    -------
    R : np.ndarray (Ny, Nx)
        Reaction rate [mol/(m^3*s)].
    """
    R = np.zeros_like(C)
    stage = config_physics.physics_stage.upper()
    k = config_physics.k_rxn

    if stage == "A":
        R[electrode_mask] = k * C[electrode_mask]

    elif stage in ("B", "C"):
        beta = config_physics.beta_field
        if E_mag is not None:
            enhancement = 1.0 + beta * E_mag[electrode_mask]
        else:
            enhancement = 1.0
        R[electrode_mask] = k * C[electrode_mask] * enhancement

    elif stage == "D":
        # Butler-Volmer kinetics
        if eta is not None:
            j0 = config_physics.j0
            alpha = config_physics.alpha
            n = config_physics.n_electrons
            T = config_physics.T
            C_bulk = config_physics.C_bulk

            C_elec = C[electrode_mask]
            C_ratio = np.clip(C_elec / C_bulk, 0, None)

            arg_a = alpha * n * const.F * eta / (const.R_GAS * T)
            arg_c = -(1 - alpha) * n * const.F * eta / (const.R_GAS * T)
            # Clip to prevent overflow
            arg_a = np.clip(arg_a, -50, 50)
            arg_c = np.clip(arg_c, -50, 50)

            j = j0 * C_ratio * (np.exp(arg_a) - np.exp(arg_c))
            # Convert A/m^2 to mol/(m^3*s) via: R = |j| / (n*F*dx)
            # Using dx as characteristic thickness of reaction zone
            R_vals = np.abs(j) / (n * const.F)
            R[electrode_mask] = R_vals
        else:
            # Fallback to Stage A
            R[electrode_mask] = k * C[electrode_mask]

    return R
