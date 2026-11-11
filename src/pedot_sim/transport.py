"""
Mass transport solver for EDOT monomer concentration.

Stage A: Pure diffusion-reaction
    dC/dt = D * nabla^2(C) - R

Stage C: Nernst-Planck (diffusion + migration)
    dC/dt = nabla . (D*nabla(C) + (z*F*D)/(R_gas*T) * C * nabla(V)) - R

Uses conservative masked theta diffusion coupled to implicit linear reaction.
Migration and concentration-dependent kinetic coefficients are split in time;
overall time accuracy is first order.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from . import constants as const
from .config import PhysicsConfig

logger = logging.getLogger(__name__)


def _diffusion_operator(grid, domain_mask):
    """Conservative finite volumes on uniform or graded Cartesian grids."""
    shape = domain_mask.shape
    ids = np.arange(domain_mask.size).reshape(shape)
    rows, cols, vals = [], [], []
    for axis, spacing in [(0, grid.dy), (1, grid.dx)]:
        n = shape[axis]
        coords = getattr(grid, "y" if axis == 0 else "x", np.arange(n)*spacing)
        widths = getattr(grid, "widths_y" if axis == 0 else "widths_x", np.full(n, spacing))
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        active = domain_mask[a] & domain_mask[b]
        left, right = ids[a][active], ids[b][active]
        resh = (n-1, 1) if axis == 0 else (1, n-1)
        for i, j, w in [(left, right, widths[:-1]), (right, left, widths[1:])]:
            rates = np.broadcast_to((1/(w*np.diff(coords))).reshape(resh), active.shape)[active]
            rows.extend(i); cols.extend(j); vals.extend(rates)
            rows.extend(i); cols.extend(i); vals.extend(-rates)
    return sparse.coo_matrix((vals, (rows, cols)), shape=(ids.size, ids.size)).tocsr()


from functools import lru_cache
from scipy.sparse.linalg import factorized


@lru_cache(maxsize=8)
def _diffusion_system(shape, mask_bytes, feeder_bytes, x, y, wx, wy, D, dt, theta, reaction_bytes):
    from types import SimpleNamespace
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    feeder = np.frombuffer(feeder_bytes, dtype=bool).reshape(shape)
    grid = SimpleNamespace(x=np.array(x), y=np.array(y), widths_x=np.array(wx),
                           widths_y=np.array(wy), dx=min(wx), dy=min(wy))
    L = D * _diffusion_operator(grid, mask)
    I = sparse.eye(mask.size, format="csr")
    reaction = np.frombuffer(reaction_bytes, dtype=float)
    lhs = (I - theta * dt * L + sparse.diags(dt*reaction)).tolil()
    for i in np.flatnonzero(feeder | ~mask):
        lhs.rows[i] = [int(i)]; lhs.data[i] = [1.0]
    return factorized(lhs.tocsc()), I + (1-theta) * dt * L


def solve_diffusion_step(C, grid, D, dt, R_sink, domain_mask, C_bulk,
                         left_feeder, right_feeder, theta=0.5, reaction_coefficient=None):
    """Theta diffusion solve with reflecting walls and fixed feeder reservoirs.

    Nonnegative linear reaction is solved simultaneously with diffusion by backward Euler.
    The orchestrator accounts for dt*reaction_coefficient*C_new as consumed material.
    """
    from .utils_grid import cell_widths
    wx, wy = cell_widths(grid)
    if C.shape != domain_mask.shape or np.any(C < 0) or not np.isfinite(C).all():
        raise ValueError("Concentration must be finite, nonnegative and match the domain")
    if np.any(R_sink < 0) or not np.isfinite(R_sink).all():
        raise ValueError("Reaction sink must be finite and nonnegative")
    if D < 0 or dt <= 0 or C_bulk < 0 or not 0 <= theta <= 1:
        raise ValueError("Invalid diffusion parameters")
    feeder = left_feeder | right_feeder
    fourier = D * dt * (grid.dx**-2 + grid.dy**-2)
    if theta < 0.5 and fourier > 0.5 / (1 - 2*theta):
        raise ValueError("Unstable explicit diffusion timestep")
    coefficient = (np.divide(R_sink, C, out=np.zeros_like(C), where=C>0)
                   if reaction_coefficient is None else np.asarray(reaction_coefficient, dtype=float))
    if coefficient.shape != C.shape or np.any(coefficient < 0) or not np.isfinite(coefficient).all():
        raise ValueError("Reaction coefficients must be finite, nonnegative and match C")
    solve, rhs_op = _diffusion_system(C.shape, domain_mask.tobytes(),
        feeder.tobytes(), tuple(grid.x), tuple(grid.y), tuple(wx), tuple(wy), D, float(f"{dt:.12g}"), theta, coefficient.ravel().tobytes())
    rhs = rhs_op @ np.where(domain_mask, C, 0.).ravel()
    rhs[feeder.ravel()] = C_bulk; rhs[~domain_mask.ravel()] = 0.
    result = solve(rhs).reshape(C.shape)
    if not np.isfinite(result).all() or result.min() < -1e-10 * max(1., C_bulk):
        raise ValueError("Diffusion lost positivity; reduce dt or use theta=1")
    result = np.maximum(result, 0.)
    result[feeder] = C_bulk
    return result


def add_migration_flux(C, V, grid, D, z, dt, domain_mask, T=const.T_DEFAULT, cfl_safety=0.8):
    """Conservative upwind face flux, subcycled for positivity, insulated walls.

    This step has closed external faces; the orchestrator resets reservoir cells.
    """
    from .utils_grid import cell_widths, cell_areas
    if T <= 0 or D < 0 or dt <= 0 or not 0 < cfl_safety <= 1:
        raise ValueError("Invalid migration parameters")
    if not np.isfinite(C).all() or not np.isfinite(V).all() or np.any(C < 0):
        raise ValueError("Migration needs finite fields and nonnegative concentration")
    mobility = z * const.F * D / (const.R_GAS * T)
    result = np.where(domain_mask, C, 0.).copy()
    wx, wy = cell_widths(grid)
    volumes = cell_areas(grid)
    faces = []; outgoing = np.zeros_like(C)
    for axis, coords, widths, face_widths in [(0, grid.y, wy, wx), (1, grid.x, wx, wy)]:
        a = [slice(None), slice(None)]; b = a.copy()
        a[axis] = slice(None, -1); b[axis] = slice(1, None)
        a, b = tuple(a), tuple(b)
        reshape = (-1, 1) if axis == 0 else (1, -1)
        distance = np.diff(coords).reshape(reshape)
        face = face_widths.reshape((1, -1) if axis == 0 else (-1, 1))
        velocity = -mobility*(V[b]-V[a])/distance
        velocity = np.where(domain_mask[a] & domain_mask[b], velocity, 0.)
        outgoing[a] += np.maximum(velocity, 0.)*face/volumes[a]
        outgoing[b] += np.maximum(-velocity, 0.)*face/volumes[b]
        faces.append((a, b, face, velocity))
    n = max(1, int(np.ceil(dt*outgoing.max()/cfl_safety)))
    if n > 100000:
        raise ValueError("Migration requires excessive substeps; reduce dt")
    for _ in range(n):
        delta = np.zeros_like(C)
        for a, b, face, velocity in faces:
            transfer = dt/n*face*velocity*np.where(velocity>=0, result[a], result[b])
            delta[a] -= transfer/volumes[a]; delta[b] += transfer/volumes[b]
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


def reservoir_exchange(C_old, C_new, grid, D, dt, theta, domain_mask, feeders):
    """Independent diffusive boundary flux integral [mol/m], plus imposed-cell reset."""
    from .utils_grid import cell_widths, cell_areas
    wx, wy = cell_widths(grid)
    midpoint = theta*C_new+(1-theta)*C_old
    amount = float(np.sum((C_new-C_old)[feeders]*cell_areas(grid)[feeders]))
    for axis, coords, facewidth in [(0, grid.y, wx), (1, grid.x, wy)]:
        a=[slice(None),slice(None)];b=a.copy()
        a[axis]=slice(None,-1);b[axis]=slice(1,None);a,b=tuple(a),tuple(b)
        distance=np.diff(coords).reshape((-1,1) if axis==0 else (1,-1))
        face=facewidth.reshape((1,-1) if axis==0 else (-1,1))
        active=domain_mask[a]&domain_mask[b]
        sign=feeders[a].astype(int)-feeders[b].astype(int)
        flux=D*dt*(midpoint[a]-midpoint[b])*face/distance
        amount += float(np.sum(np.where(active, sign*flux, 0.)))
    return amount
