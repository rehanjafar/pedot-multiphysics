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

from .config import SimConfig
from .field import (
    compute_electric_field,
    compute_field_magnitude,
    get_overpotential_at_bpe,
    fixed_domain_potential,
)
from .geometry import DomainInfo
from .nucleation import NucleationModel
from .transport import add_migration_flux, compute_reaction_rate, solve_diffusion_step, reservoir_exchange
from .utils_grid import cell_areas, electrode_weights
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
    volumes = cell_areas(grid)
    widths = electrode_weights(grid, domain.electrode_mask)
    heights = volumes[domain.electrode_mask]/widths
    phy = config.physics
    stage = phy.physics_stage.upper()
    solver_info: dict = {}

    # 1. Compute waveform at current time
    waveform_func = get_waveform(config.waveform)
    V_applied = float(waveform_func(t + dt/2))
    V_left, V_right = get_boundary_potentials(V_applied)

    # 2. Solve electric field (Stage >= B)
    V_field: Optional[np.ndarray] = None
    E_mag: Optional[np.ndarray] = None
    eta: Optional[np.ndarray] = None

    if stage in ("B", "C", "D"):
        V_field, n_iter, residual = fixed_domain_potential(
            domain, V_applied, config.numerical.laplace_tol,
            config.numerical.laplace_max_iter,
            abs(config.waveform.V_DC)+config.waveform.V_AC)
        if residual >= config.numerical.laplace_tol:
            raise RuntimeError("Laplace solver did not converge")
        Ex, Ey = compute_electric_field(V_field, grid, domain.domain_mask)
        E_mag = compute_field_magnitude(Ex, Ey)
        solver_info["laplace_iters"] = n_iter
        solver_info["laplace_residual"] = residual


    # Freeze field/BV coefficients over this step, then solve diffusion and the
    # linear concentration-dependent reaction together. Splitting a stiff surface
    # sink from diffusion makes results depend spuriously on cell height.
    if stage == "D":
        eta = get_overpotential_at_bpe(V_field, domain.electrode_mask, C, phy, widths)
    reference_C = np.where(domain.domain_mask, phy.C_bulk, 0.)
    reference_flux = compute_reaction_rate(reference_C, domain.electrode_mask, phy, E_mag, eta)
    velocity = reference_flux[domain.electrode_mask]/phy.C_bulk
    if nucleation_model is not None:
        velocity *= nucleation_model.flux_multiplier(dt, widths)
    coefficient = np.zeros_like(C)
    coefficient[domain.electrode_mask] = velocity/heights
    C_new = solve_diffusion_step(C, grid, phy.diffusion_coefficient, dt,
        np.zeros_like(C), domain.domain_mask, phy.C_bulk, domain.left_feeder_mask,
        domain.right_feeder_mask, config.numerical.implicit_theta,
        reaction_coefficient=coefficient)
    R_field = coefficient*C_new
    R_at_electrode = velocity*C_new[domain.electrode_mask]
    consumed = dt*R_field[domain.electrode_mask]
    feeders = domain.left_feeder_mask | domain.right_feeder_mask
    exchange = reservoir_exchange(C, C_new, grid, phy.diffusion_coefficient, dt,
        config.numerical.implicit_theta, domain.domain_mask, feeders)
    if stage in ("C", "D") and V_field is not None:
        C_new = add_migration_flux(C_new, V_field, grid, phy.diffusion_coefficient,
            phy.z_species, dt, domain.domain_mask, T=phy.T, cfl_safety=config.numerical.cfl_safety)
        exchange += float(np.sum((phy.C_bulk-C_new[feeders])*volumes[feeders]))
        C_new[feeders] = phy.C_bulk
    dhdt = (R_at_electrode * phy.deposited_molar_mass
            * phy.deposition_efficiency / phy.film_density)
    h_new = h + dt * dhdt
    if not np.isfinite(C_new).all() or not np.isfinite(h_new).all():
        raise ValueError("Nonfinite simulation state")

    solver_info["reservoir_exchange_mol_per_m"] = exchange
    solver_info["consumed_mol_per_m"] = float(np.sum(consumed*volumes[domain.electrode_mask]))
    solver_info["applied_voltage_midpoint"] = V_applied

    # Compute flux diagnostics
    mean_flux = float(np.average(R_at_electrode, weights=widths)) if len(R_at_electrode) > 0 else 0.0
    total_flux = float(np.sum(R_at_electrode*widths)) if len(R_at_electrode) > 0 else 0.0
    growth_rate_val = float(np.average(dhdt, weights=widths)) if len(dhdt) > 0 else 0.0

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
