"""
Mass transport solver for EDOT monomer concentration.

Stage A: Pure diffusion-reaction
    dC/dt = D * nabla^2(C) - R

Stage C: Nernst-Planck (diffusion + migration)
    dC/dt = nabla . (D*nabla(C) + (z*F*D)/(R_gas*T) * C * nabla(V)) - R

Uses conservative masked theta diffusion, upwind migration and local implicit reaction splitting. Overall splitting is first order in time.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from . import constants as const
from .utils_grid import Grid2D

logger = logging.getLogger(__name__)


def _diffusion_operator(grid, domain_mask):
    """Cell-centred finite-volume graph Laplacian; insulated inactive faces."""
    shape = domain_mask.shape
    ids = np.arange(domain_mask.size).reshape(shape)
    rows, cols, vals = [], [], []
    for axis, spacing in [(0, grid.dy), (1, grid.dx)]:
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        active = domain_mask[a] & domain_mask[b]
        left, right = ids[a][active], ids[b][active]
        for i, j in [(left, right), (right, left)]:
            rows.extend(i); cols.extend(j); vals.extend(np.full(len(i), 1 / spacing**2))
            rows.extend(i); cols.extend(i); vals.extend(np.full(len(i), -1 / spacing**2))
    return sparse.coo_matrix((vals, (rows, cols)), shape=(ids.size, ids.size)).tocsr()


from functools import lru_cache
from scipy.sparse.linalg import factorized


@lru_cache(maxsize=16)
def _diffusion_system(shape, mask_bytes, feeder_bytes, dx, dy, D, dt, theta):
    from types import SimpleNamespace
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    feeder = np.frombuffer(feeder_bytes, dtype=bool).reshape(shape)
    L = D * _diffusion_operator(SimpleNamespace(dx=dx, dy=dy), mask)
    I = sparse.eye(mask.size, format="csr")
    lhs = (I - theta * dt * L).tolil()
    for i in np.flatnonzero(feeder | ~mask):
        lhs.rows[i] = [int(i)]; lhs.data[i] = [1.0]
    return factorized(lhs.tocsc()), I + (1-theta) * dt * L


def solve_diffusion_step(C, grid, D, dt, R_sink, domain_mask, C_bulk,
                         left_feeder, right_feeder, theta=0.5):
    """Theta diffusion solve with reflecting walls and fixed feeder reservoirs.

    Nonnegative reaction is integrated locally by backward Euler after diffusion.
    The production growth orchestrator handles reaction separately to account for
    exactly the same consumed material in the deposited film.
    """
    feeder = left_feeder | right_feeder
    fourier = D * dt * (grid.dx**-2 + grid.dy**-2)
    if theta < 0.5 and fourier > 0.5 / (1 - 2*theta):
        raise ValueError("Unstable explicit diffusion timestep")
    solve, rhs_op = _diffusion_system(C.shape, domain_mask.tobytes(),
        feeder.tobytes(), grid.dx, grid.dy, D, dt, theta)
    rhs = rhs_op @ np.where(domain_mask, C, 0.).ravel()
    rhs[feeder.ravel()] = C_bulk; rhs[~domain_mask.ravel()] = 0.
    result = solve(rhs).reshape(C.shape)
    if not np.isfinite(result).all() or result.min() < -1e-10 * max(1., C_bulk):
        raise ValueError("Diffusion lost positivity; reduce dt or use theta=1")
    result = np.maximum(result, 0.)
    coefficient = np.divide(R_sink, C, out=np.zeros_like(C), where=C > 0)
    result /= 1 + dt * coefficient
    result[feeder] = C_bulk
    return result


def add_migration_flux(C, V, grid, D, z, dt, domain_mask, T=const.T_DEFAULT, cfl_safety=0.8):
    """Conservative upwind face flux, subcycled for positivity, insulated walls.

    This step has closed external faces; the orchestrator resets reservoir cells.
    """
    mobility = z * const.F * D / (const.R_GAS * T)
    result = np.where(domain_mask, C, 0.).copy()
    faces = []; outgoing = np.zeros_like(C)
    for axis, spacing in [(0, grid.dy), (1, grid.dx)]:
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        velocity = -mobility * (V[b] - V[a]) / spacing
        velocity = np.where(domain_mask[a] & domain_mask[b], velocity, 0.)
        outgoing[a] += np.maximum(velocity, 0.) / spacing
        outgoing[b] += np.maximum(-velocity, 0.) / spacing
        faces.append((a, b, spacing, velocity))
    n = max(1, int(np.ceil(dt * outgoing.max() / cfl_safety)))
    for _ in range(n):
        delta = np.zeros_like(C)
        for a, b, spacing, velocity in faces:
            transfer = dt/n/spacing * velocity * np.where(velocity >= 0, result[a], result[b])
            delta[a] -= transfer; delta[b] += transfer
        result += delta
    if not np.isfinite(result).all() or result.min() < -1e-10:
        raise ValueError("Invalid migration concentration")
    return np.maximum(result, 0.)


def compute_reaction_rate(
    C: np.ndarray,
    electrode_mask: np.ndarray,
    config_physics: "PhysicsConfig",
    E_mag: np.ndarray | None = None,
    eta: np.ndarray | None = None,
) -> np.ndarray:
    """Compute surface monomer flux [mol/(m² s)], positive for deposition.

    Stage A: R = k_rxn * C  (at electrode only)
    Stage B: R = k_rxn * C * (1 + beta*|E|)  (field-enhanced)
    Stage C: same as B (migration handled in transport)
    Stage D: flux = max(j_BV, 0) / (n*F)  (Butler-Volmer based)

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
        Surface flux [mol/(m^2*s)].
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
            raise ValueError("Stages B/C require the electric field")
        R[electrode_mask] = k * C[electrode_mask] * enhancement

    elif stage == "D":
        if eta is None:
            raise ValueError("Stage D requires electrode overpotential")
        from .kinetics import butler_volmer_flux, flux_to_consumption_rate
        j = butler_volmer_flux(eta, config_physics.j0, C[electrode_mask],
                              config_physics.C_bulk, config_physics.alpha,
                              config_physics.n_electrons, config_physics.T)
        R[electrode_mask] = flux_to_consumption_rate(j, config_physics.n_electrons)
    else:
        raise ValueError(f"Unknown physics stage: {stage}")
    return R
