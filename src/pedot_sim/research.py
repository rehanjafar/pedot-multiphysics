"""Dimensioned process diagnostics and explicit limits of the fixed-strip model.

All extensive quantities are per metre of out-of-plane depth. These are model
accounts, not measured process yield, cell current, or a lifecycle assessment.
"""
from __future__ import annotations

import numpy as np

from . import constants
from .utils_grid import cell_widths, cell_areas, electrode_weights


METRIC_UNITS = {
    "mean_h": "m", "max_h": "m", "UI": "m", "AI": "1",
    "thickness_cv": "1", "left_right_asymmetry": "1",
    "depletion_depth": "m", "mean_growth_rate": "m/s",
    "consumed_mol_per_m": "mol/m", "retained_repeat_mol_per_m": "mol/m",
    "deposited_mass_kg_per_m": "kg/m", "equivalent_charge_C_per_m": "C/m",
    "reservoir_exchange_mol_per_m": "mol/m", "mass_balance_error_mol_per_m": "mol/m",
    "electrolyte_ohmic_energy_J_per_m": "J/m",
    "electrolyte_energy_per_deposited_mass_J_per_kg": "J/kg",
    "diffusion_length_per_period": "m", "cells_per_diffusion_length": "1",
}


def ohmic_power_per_depth(V, domain, conductivity):
    """Sum sigma*(potential jump)^2*face-area/distance over active edges [W/m].

    Uses the same discrete domain as the diffusion/Laplace graph. Excludes
    interfacial polarization, the power supply, heating feedback and reactions.
    """
    if V is None:
        return None
    mask, g = domain.domain_mask, domain.grid
    wx, wy = cell_widths(g)
    x = (mask[:, 1:] & mask[:, :-1])*np.diff(V, axis=1)**2*wy[:, None]/np.diff(g.x)[None, :]
    y = (mask[1:, :] & mask[:-1, :])*np.diff(V, axis=0)**2*wx[None, :]/np.diff(g.y)[:, None]
    return float(conductivity * (x.sum()+y.sum()))


def process_diagnostics(config, domain, initial_C, final_C, h, consumed, reservoir, energy):
    """Audit the complete discrete material balance and summarize process proxies."""
    ph, grid = config.physics, domain.grid
    volumes = cell_areas(grid)
    weights = electrode_weights(grid, domain.electrode_mask)
    initial = float(np.sum(initial_C*volumes))
    remaining = float(np.sum(final_C*volumes))
    retained = float(np.sum(h*weights)*ph.film_density/ph.deposited_molar_mass)
    error = initial + reservoir - remaining - consumed
    film_error = retained - consumed*ph.deposition_efficiency
    tolerance = 1e-10 * max(initial, abs(reservoir), consumed, 1e-30)
    if abs(error) > tolerance or abs(film_error) > tolerance:
        raise ValueError("Material balance failed; no manufacturing conclusions may be drawn")
    mass = retained*ph.deposited_molar_mass
    mean = float(np.average(h, weights=weights))
    half = len(h)//2
    left, right = float(np.sum(h[:half]*weights[:half])), float(np.sum(h[-half:]*weights[-half:]))
    diffusion_length = float(np.sqrt(ph.diffusion_coefficient/config.waveform.frequency))
    metrics = {
        "initial_inventory_mol_per_m": initial,
        "remaining_inventory_mol_per_m": remaining,
        "consumed_mol_per_m": consumed,
        "reservoir_exchange_mol_per_m": reservoir,
        "retained_repeat_mol_per_m": retained,
        "deposited_mass_kg_per_m": mass,
        "unretained_mol_per_m": consumed-retained,
        "mass_balance_error_mol_per_m": error,
        "film_balance_error_mol_per_m": film_error,
        "equivalent_charge_C_per_m": consumed*ph.n_electrons*constants.F,
        "thickness_cv": float(np.sqrt(np.average((h-mean)**2,weights=weights))/mean) if mean > 0 else None,
        "left_right_asymmetry": (right-left)/float(np.sum(h*weights)) if np.sum(h)>0 else None,
        "electrolyte_ohmic_energy_J_per_m": energy,
        "electrolyte_energy_per_deposited_mass_J_per_kg": energy/mass if energy is not None and mass>0 else None,
        "diffusion_length_per_period": diffusion_length,
        "cells_per_diffusion_length": diffusion_length/grid.dy,
        "resolved_electrode_length": float(np.sum(weights)),
    }
    warnings = [
        "Fixed strip and fixed electrolyte domain: no lateral fronts, dendrites, branching or geometry feedback.",
        "Uncalibrated parameters; no experimental validation dataset has been supplied.",
        "No explicit dopant, counter-reaction, double-layer capacitance, convection, heat or side reactions.",
        "Ohmic energy and equivalent charge are proxies; actual cell energy, current and sustainability are not predicted.",
        "Film conductivity, adhesion, porosity, doping, optical and thermoelectric properties require measurements.",
        "Viscosity and ionic_strength are metadata; diffusion and conductivity are independent input parameters.",
    ]
    if ph.physics_stage == "A":
        warnings.append("Stage A has no voltage dependence; it is a diffusion-reaction benchmark.")
    if ph.z_species == 0 and ph.physics_stage in ("C", "D"):
        warnings.append("Neutral EDOT has no migration term. Charged intermediates are not represented.")
    if ph.z_species != 0:
        warnings.append("Nonzero species charge represents an effective charged species, not neutral EDOT.")
    if diffusion_length < 3*grid.dy:
        warnings.append("Fewer than three cells per sqrt(D/f): waveform-scale diffusion is unresolved; refine the grid.")
    if config.nucleation.enable_noise:
        warnings.append("Stochastic rate multiplier is exploratory and has no calibrated nucleation statistics.")
    if float(np.max(h)) > 0.1*grid.dy:
        warnings.append("Thickness exceeds 10% of a cell: fixed-geometry approximation needs reassessment.")
    audit = {
        "model": "fixed-domain, weak-coupling surface-deposition approximation",
        "experimental_validation": "not_established",
        "numerical_convergence": "requires_separate_mesh_and_timestep_studies",
        "material_balance_passed": True,
        "units": METRIC_UNITS,
        "warnings": warnings,
        "inputs_requiring_calibration": ["diffusion_coefficient", "k_rxn", "j0", "alpha", "beta_field", "film_density", "deposition_efficiency", "n_electrons"],
        "scientific_basis": "docs/SCIENTIFIC_BASIS.md",
    }
    return metrics, audit
