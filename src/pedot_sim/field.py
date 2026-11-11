"""
Electric field solver for Stage B and above.

Solves the Laplace equation in the electrolyte domain:
    nabla^2 V = 0

with Dirichlet BCs on feeder electrodes and Neumann (no-flux)
on insulating walls.

Then computes:
    E = -nabla V
    |E| = sqrt(Ex^2 + Ey^2)

Implementation uses Red-Black Gauss-Seidel SOR with vectorized
half-sweeps.  Neumann BC at domain boundaries is handled via
ghost-cell mirroring: out-of-domain neighbors are replaced by
the cell's own value before the stencil is evaluated, which sets
the normal gradient to zero.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .utils_grid import Grid2D

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Laplace solver
# ---------------------------------------------------------------------------

def solve_laplace_2d(
    grid: Grid2D,
    domain_mask: np.ndarray,
    V_left: float,
    V_right: float,
    left_feeder: np.ndarray,
    right_feeder: np.ndarray,
    wall_mask: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 10000,
    omega: Optional[float] = None,
    V_init: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, int, float]:
    """Solve Laplace equation nabla^2 V = 0 using Red-Black SOR.

    Uses Gauss-Seidel ordering (red-black checkerboard) so that
    the optimal SOR relaxation parameter works correctly.
    Neumann BC (dV/dn = 0) on domain boundaries is enforced
    implicitly via ghost-cell mirroring in the stencil.

    Parameters
    ----------
    grid : Grid2D
    domain_mask : np.ndarray (Ny, Nx)
        True for active cells.
    V_left, V_right : float
        Dirichlet BC on left/right feeder electrodes [V].
    left_feeder, right_feeder : np.ndarray (Ny, Nx)
        Boolean masks for feeder electrode locations.
    wall_mask : np.ndarray (Ny, Nx)
        Boolean mask for insulating walls (kept for API compat;
        Neumann BC is now handled automatically by the ghost-cell
        stencil for all domain-boundary cells).
    tol : float
        Convergence tolerance on max absolute change per iteration.
    max_iter : int
        Maximum iterations (one iteration = red sweep + black sweep).
    omega : float or None
        SOR relaxation parameter.  If *None*, the optimal value
        ``2 / (1 + sin(pi / N))`` is computed automatically.
    V_init : np.ndarray, optional
        Initial guess (warm-start from previous time step).

    Returns
    -------
    V : np.ndarray (Ny, Nx)
        Potential field [V].
    n_iter : int
        Iterations used.
    residual : float
        Final convergence metric (max |delta V|).
    """
    Ny, Nx = domain_mask.shape
    dx2 = grid.dx ** 2
    dy2 = grid.dy ** 2
    inv_denom = 1.0 / (2.0 * (1.0 / dx2 + 1.0 / dy2))

    # ---- Optimal omega for Red-Black Gauss-Seidel SOR -------------------
    if omega is None:
        N_eff = max(Ny, Nx)
        omega = 2.0 / (1.0 + np.sin(np.pi / N_eff))

    # ---- Initialise V ---------------------------------------------------
    if V_init is not None:
        V = V_init.copy().astype(np.float64)
    else:
        V = np.zeros((Ny, Nx), dtype=np.float64)
        x_frac = np.linspace(0.0, 1.0, Nx)
        V[:] = V_left + (V_right - V_left) * x_frac[np.newaxis, :]

    V[left_feeder] = V_left
    V[right_feeder] = V_right
    V[~domain_mask] = 0.0

    # ---- Masks ----------------------------------------------------------
    updatable = domain_mask & (~left_feeder) & (~right_feeder)

    if not np.any(updatable):
        return V, 0, 0.0

    # Red-black checkerboard partitioning
    parity = (np.arange(Ny)[:, None] + np.arange(Nx)[None, :]) % 2
    red = updatable & (parity == 0)
    black = updatable & (parity == 1)

    # ---- Iterate --------------------------------------------------------
    residual = float("inf")

    for iteration in range(max_iter):
        # Red half-sweep (updates red cells using current black values)
        max_delta_r = _gs_sor_half_sweep(
            V, domain_mask, red, dx2, dy2, inv_denom, omega,
        )

        # Black half-sweep (updates black cells using updated red values)
        max_delta_b = _gs_sor_half_sweep(
            V, domain_mask, black, dx2, dy2, inv_denom, omega,
        )

        residual = max(max_delta_r, max_delta_b)

        if residual < tol:
            logger.debug(
                "Laplace converged in %d iters (res=%.2e, omega=%.3f)",
                iteration + 1, residual, omega,
            )
            return V, iteration + 1, residual

    logger.warning(
        "Laplace did NOT converge: %d iters, residual=%.2e", max_iter, residual,
    )
    return V, max_iter, residual


def _gs_sor_half_sweep(
    V: np.ndarray,
    domain_mask: np.ndarray,
    sweep_mask: np.ndarray,
    dx2: float,
    dy2: float,
    inv_denom: float,
    omega: float,
) -> float:
    """One half-sweep of Red-Black Gauss-Seidel SOR, in-place.

    For each cell in *sweep_mask*, the standard 5-point Laplace
    stencil is evaluated.  Neighbors that lie outside *domain_mask*
    are replaced by the cell's own value (Neumann ghost-cell BC,
    i.e. dV/dn = 0 at the domain boundary).

    Parameters
    ----------
    V : np.ndarray (Ny, Nx)
        Potential field, **modified in-place**.
    domain_mask : bool array (Ny, Nx)
    sweep_mask : bool array (Ny, Nx)
        Cells to update in this half-sweep.
    dx2, dy2 : float
        Squared grid spacings.
    inv_denom : float
        ``1 / (2*(1/dx2 + 1/dy2))``.
    omega : float
        SOR relaxation factor.

    Returns
    -------
    float
        Maximum absolute correction applied.
    """
    Ny, Nx = V.shape

    # -- Build effective neighbor values with Neumann ghost mirroring ------
    # East: V[j, i+1] if in domain, else V[j, i] (mirror)
    V_E = np.empty_like(V)
    V_E[:, :-1] = np.where(domain_mask[:, 1:], V[:, 1:], V[:, :-1])
    V_E[:, -1] = V[:, -1]  # grid boundary → Neumann

    # West: V[j, i-1]
    V_W = np.empty_like(V)
    V_W[:, 1:] = np.where(domain_mask[:, :-1], V[:, :-1], V[:, 1:])
    V_W[:, 0] = V[:, 0]

    # South: V[j+1, i]
    V_S = np.empty_like(V)
    V_S[:-1, :] = np.where(domain_mask[1:, :], V[1:, :], V[:-1, :])
    V_S[-1, :] = V[-1, :]

    # North: V[j-1, i]
    V_N = np.empty_like(V)
    V_N[1:, :] = np.where(domain_mask[:-1, :], V[:-1, :], V[1:, :])
    V_N[0, :] = V[0, :]

    # -- Gauss-Seidel target -----------------------------------------------
    V_target = ((V_E + V_W) / dx2 + (V_N + V_S) / dy2) * inv_denom

    # -- SOR update: V += omega * (V_target - V) ---------------------------
    correction = omega * (V_target[sweep_mask] - V[sweep_mask])
    V[sweep_mask] += correction

    if correction.size == 0:
        return 0.0
    return float(np.max(np.abs(correction)))


# ---------------------------------------------------------------------------
# Electric field and overpotential
# ---------------------------------------------------------------------------

def compute_electric_field(V: np.ndarray, grid: Grid2D, domain_mask=None) -> tuple[np.ndarray, np.ndarray]:
    """Compute electric field E = -grad(V) using central differences.

    Parameters
    ----------
    V : np.ndarray (Ny, Nx)
        Potential field [V].
    grid : Grid2D

    Returns
    -------
    Ex, Ey : np.ndarray (Ny, Nx)
        Electric field components [V/m].
    """
    if domain_mask is None:
        Ey, Ex = np.gradient(V, grid.dy, grid.dx)
        return -Ex, -Ey
    components = []
    for axis, spacing in [(1, grid.dx), (0, grid.dy)]:
        plus = np.roll(V, -1, axis); minus = np.roll(V, 1, axis)
        mp = np.roll(domain_mask, -1, axis); mm = np.roll(domain_mask, 1, axis)
        edge = [slice(None), slice(None)]; edge[axis] = -1; mp[tuple(edge)] = False
        edge[axis] = 0; mm[tuple(edge)] = False
        grad = (np.where(mp, plus, V) - np.where(mm, minus, V)) / (2*spacing)
        components.append(np.where(domain_mask, -grad, 0.))
    return tuple(components)


def compute_field_magnitude(Ex: np.ndarray, Ey: np.ndarray) -> np.ndarray:
    """Compute |E| = sqrt(Ex^2 + Ey^2) [V/m]."""
    return np.sqrt(Ex ** 2 + Ey ** 2)


def get_overpotential_at_bpe(
    V: np.ndarray,
    electrode_mask: np.ndarray,
    concentration=None, physics=None,
) -> np.ndarray:
    """Balance signed phenomenological BV current with a uniform metal potential.

    eta = metal - solution (zero reference equilibrium). This is a weak-coupling
    proxy; it does not solve interfacial polarization of the electrolyte or a
    calibrated cathodic counter-reaction, so cannot predict cell current.
    """
    from scipy.optimize import brentq
    from .kinetics import butler_volmer_flux
    values = V[electrode_mask]
    if values.size == 0:
        raise ValueError("Electrode has no cells")
    if concentration is None or physics is None:
        raise ValueError("Floating potential requires concentration and kinetics")
    weights = concentration[electrode_mask]
    if not np.any(weights > 0):
        return np.zeros_like(values)
    def net(metal):
        return float(np.sum(butler_volmer_flux(metal-values, physics.j0,
            weights, physics.C_bulk, physics.alpha, physics.n_electrons, physics.T)))
    metal = brentq(net, float(values.min()-1.), float(values.max()+1.), xtol=1e-13)
    return metal-values


from functools import lru_cache


@lru_cache(maxsize=8)
def _unit_potential(shape, mask_bytes, left_bytes, right_bytes, dx, dy, tol, max_iter):
    from types import SimpleNamespace
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    left = np.frombuffer(left_bytes, dtype=bool).reshape(shape)
    right = np.frombuffer(right_bytes, dtype=bool).reshape(shape)
    grid = SimpleNamespace(dx=dx, dy=dy)
    V, iterations, residual = solve_laplace_2d(
        grid, mask, .5, -.5, left, right, np.zeros(shape, dtype=bool),
        tol=tol, max_iter=max_iter)
    if residual >= tol:
        raise RuntimeError("Laplace solver did not converge")
    V.setflags(write=False)
    return V, iterations, residual


def fixed_domain_potential(domain, applied_voltage, tol, max_iter, voltage_bound):
    """Scale a cached unit-drive solution without weakening absolute tolerance."""
    unit_tol = tol/max(1., voltage_bound)
    V, iterations, residual = _unit_potential(
        domain.domain_mask.shape, domain.domain_mask.tobytes(),
        domain.left_feeder_mask.tobytes(), domain.right_feeder_mask.tobytes(),
        domain.grid.dx, domain.grid.dy, unit_tol, max_iter)
    return V*applied_voltage, iterations, residual*abs(applied_voltage)
