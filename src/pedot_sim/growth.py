"""
Film growth evolution and master timestep orchestrator.

Manages the coupled physics: at each timestep,
1. Compute waveform V(t) -> boundary potentials
2. (Stage >= B) Solve Laplace for electric potential
3. Compute reaction rate at electrode
4. Solve transport equation (diffusion +/- migration)
5. Update film thickness from reaction flux
6. (Optional) Add nucleation noise
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import constants as const
from .config import SimConfig
from .field import (
    compute_electric_field,
    compute_field_magnitude,
    get_overpotential_at_bpe,
    solve_laplace_2d,
)
from .geometry import DomainInfo
from .kinetics import butler_volmer_flux, flux_to_growth_rate
from .nucleation import NucleationModel
from .transport import add_migration_flux, compute_reaction_rate, solve_diffusion_step
from .utils_grid import Grid2D
from .waveforms import get_boundary_potentials, get_waveform

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of one timestep integration."""
    C: np.ndarray             # Concentration field (Ny, Nx)
    V_field: Optional[np.ndarray]  # Potential field (Ny, Nx), None for Stage A
    h: np.ndarray             # Film thickness along electrode (n_electrode,)
    mean_flux: float          # Mean reaction flux at electrode
    total_flux: float         # Total integrated flux
    growth_rate: float        # Mean dh/dt
    E_mag: Optional[np.ndarray]    # |E| field, None for Stage A
    R_field: np.ndarray       # Reaction rate field (Ny, Nx)
    solver_info: dict         # Solver diagnostics


def evolve_one_timestep(
    C: np.ndarray,
    h: np.ndarray,
    t: float,
    dt: float,
    config: SimConfig,
    domain: DomainInfo,
    nucleation_model: Optional[NucleationModel],
    V_prev: Optional[np.ndarray] = None,
) -> StepResult:
    """Advance the coupled system by one timestep.

    Parameters
    ----------
    C : np.ndarray (Ny, Nx)
        Current concentration field [mol/m^3].
    h : np.ndarray (n_electrode,)
        Current film thickness [m].
    t : float
        Current time [s].
    dt : float
        Time step [s].
    config : SimConfig
    domain : DomainInfo
    nucleation_model : NucleationModel, optional
    V_prev : np.ndarray, optional
        Previous potential field for warm-starting Laplace solver.

    Returns
    -------
    StepResult
    """
    grid = domain.grid
    phy = config.physics
    stage = phy.physics_stage.upper()
    solver_info: dict = {}

    # 1. Compute waveform at current time
    waveform_func = get_waveform(config.waveform)
    V_applied = float(waveform_func(t))
    V_left, V_right = get_boundary_potentials(V_applied)

    # 2. Solve electric field (Stage >= B)
    V_field: Optional[np.ndarray] = None
    E_mag: Optional[np.ndarray] = None
    eta: Optional[np.ndarray] = None

    if stage in ("B", "C", "D"):
        V_field, n_iter, residual = solve_laplace_2d(
            grid=grid,
            domain_mask=domain.domain_mask,
            V_left=V_left,
            V_right=V_right,
            left_feeder=domain.left_feeder_mask,
            right_feeder=domain.right_feeder_mask,
            wall_mask=domain.wall_mask,
            tol=config.numerical.laplace_tol,
            max_iter=config.numerical.laplace_max_iter,
            V_init=V_prev,
        )
        if residual >= config.numerical.laplace_tol:
            raise RuntimeError("Laplace solver did not converge")
        Ex, Ey = compute_electric_field(V_field, grid, domain.domain_mask)
        E_mag = compute_field_magnitude(Ex, Ey)
        solver_info["laplace_iters"] = n_iter
        solver_info["laplace_residual"] = residual

        if stage == "D":
            eta = get_overpotential_at_bpe(V_field, domain.electrode_mask, C, phy)

    # 4. Solve transport
    C_new = solve_diffusion_step(
        C=C,
        grid=grid,
        D=phy.diffusion_coefficient,
        dt=dt,
        R_sink=np.zeros_like(C),
        domain_mask=domain.domain_mask,
        C_bulk=phy.C_bulk,
        left_feeder=domain.left_feeder_mask,
        right_feeder=domain.right_feeder_mask,
        theta=config.numerical.implicit_theta,
    )

    # Stage C/D: add migration
    if stage in ("C", "D") and V_field is not None:
        C_new = add_migration_flux(
            C=C_new,
            V=V_field,
            grid=grid,
            D=phy.diffusion_coefficient,
            z=phy.z_species,
            dt=dt,
            domain_mask=domain.domain_mask,
            T=phy.T,
        )
        # Re-apply BCs
        C_new[domain.left_feeder_mask] = phy.C_bulk
        C_new[domain.right_feeder_mask] = phy.C_bulk
        C_new = np.maximum(C_new, 0.0)

    # One effective exposed interface per horizontal electrode cell:
    # area/volume = dx/(dx*dy) = 1/dy, per unit out-of-plane depth.
    if stage == "D":
        eta = get_overpotential_at_bpe(V_field, domain.electrode_mask, C_new, phy)
    surface_flux = compute_reaction_rate(C_new, domain.electrode_mask, phy, E_mag, eta)
    requested = surface_flux[domain.electrode_mask]
    if nucleation_model is not None:
        requested *= nucleation_model.flux_multiplier(dt, grid.dx)
    available = C_new[domain.electrode_mask]
    coefficient = np.divide(requested, available, out=np.zeros_like(requested), where=available > 0)
    remaining = available / (1 + dt * coefficient / grid.dy)
    consumed = available - remaining
    C_new[domain.electrode_mask] = remaining
    R_field = np.zeros_like(C)
    R_field[domain.electrode_mask] = consumed / dt
    R_at_electrode = consumed * grid.dy / dt
    dhdt = R_at_electrode * const.M_EDOT / const.RHO_PEDOT
    h_new = h + dt * dhdt
    if not np.isfinite(C_new).all() or not np.isfinite(h_new).all():
        raise ValueError("Nonfinite simulation state")

    # Compute flux diagnostics
    mean_flux = float(np.mean(R_at_electrode)) if len(R_at_electrode) > 0 else 0.0
    total_flux = float(np.sum(R_at_electrode) * grid.dx) if len(R_at_electrode) > 0 else 0.0
    growth_rate_val = float(np.mean(dhdt)) if len(dhdt) > 0 else 0.0

    return StepResult(
        C=C_new,
        V_field=V_field,
        h=h_new,
        mean_flux=mean_flux,
        total_flux=total_flux,
        growth_rate=growth_rate_val,
        E_mag=E_mag,
        R_field=R_field,
        solver_info=solver_info,
    )
