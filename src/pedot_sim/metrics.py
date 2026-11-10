"""
Morphology metrics for PEDOT film characterization.

Dependent variables (DVs) as defined in the research design:
- Anisotropy Index (AI) = h_max / h_avg
- Uniformity Index (UI) = std(h)
- Depletion zone depth
- Timescale ratio tau_diff / tau_rxn
- Growth rate time series
- Current proxy (integrated flux)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import constants as C
from .utils_grid import Grid2D


def anisotropy_index(h: np.ndarray) -> float:
    """Anisotropy Index: AI = h_max / h_avg.

    AI = 1.0 for perfectly uniform film.
    AI > 1.0 indicates preferential growth at some locations.

    Parameters
    ----------
    h : np.ndarray
        Film thickness array [m].

    Returns
    -------
    float
        Anisotropy index (dimensionless).
    """
    h_pos = h
    if len(h_pos) == 0:
        return 1.0
    h_avg = np.mean(h_pos)
    if h_avg == 0:
        return 1.0
    return float(np.max(h_pos) / h_avg)


def uniformity_index(h: np.ndarray) -> float:
    """Uniformity Index: UI = std(h).

    UI = 0 for perfectly uniform film.
    Higher UI means more non-uniform.

    Parameters
    ----------
    h : np.ndarray
        Film thickness [m].

    Returns
    -------
    float
        Standard deviation of thickness [m].
    """
    if len(h) == 0:
        return 0.0
    return float(np.std(h))


def depletion_depth(
    C_field: np.ndarray,
    C_bulk: float,
    grid: Grid2D,
    electrode_mask: np.ndarray,
    threshold: float = 0.9,
) -> float:
    """Depletion zone depth from electrode surface.

    Depth from electrode where C < threshold * C_bulk.

    Parameters
    ----------
    C_field : np.ndarray (Ny, Nx)
        Concentration field [mol/m^3].
    C_bulk : float
        Bulk concentration [mol/m^3].
    grid : Grid2D
    electrode_mask : np.ndarray (Ny, Nx)
    threshold : float
        Fraction of C_bulk defining depletion boundary.

    Returns
    -------
    float
        Depletion depth [m]. 0 if no depletion.
    """
    rows, cols = np.where(electrode_mask)
    if len(rows) == 0:
        return 0.0

    j_elec = rows[0]  # Electrode y-row index
    C_threshold = threshold * C_bulk

    # Look both above and below electrode for depletion
    max_depth = 0.0
    for direction in [-1, 1]:  # above and below
        j = j_elec + direction
        depth = 0.0
        while 0 <= j < grid.Ny:
            # Average concentration along x at this y-row
            C_row = C_field[j, cols]
            C_mean = np.mean(C_row) if len(C_row) > 0 else C_bulk
            if C_mean >= C_threshold:
                break
            depth += grid.dy
            j += direction
        max_depth = max(max_depth, depth)

    return float(max_depth)


def timescale_ratio(
    D: float,
    k_rxn: float,
    L_char: float,
) -> float:
    """Compute timescale ratio tau_diff / tau_rxn.

    tau_diff = L^2 / D      (diffusion timescale)
    tau_rxn = 1 / k_rxn     (reaction timescale, note: k_rxn has units [m/s],
                              so tau_rxn = L / k_rxn for surface reaction
                              or simply 1/k_rxn if k_rxn is [1/s])

    For a heterogeneous rate constant k [m/s] and characteristic length L:
        tau_diff = L^2 / D
        tau_rxn = L / k

    Ratio = tau_diff / tau_rxn = (L^2/D) / (L/k) = L*k/D

    Parameters
    ----------
    D : float
        Diffusion coefficient [m^2/s].
    k_rxn : float
        Reaction rate constant [m/s].
    L_char : float
        Characteristic length (e.g., electrode length) [m].

    Returns
    -------
    float
        Timescale ratio (dimensionless).
        > 1 means diffusion-limited (anisotropic growth expected).
        < 1 means reaction-limited (uniform growth expected).
    """
    if D == 0:
        return float("inf")
    return L_char * k_rxn / D


def mean_growth_rate(
    h_history: np.ndarray,
    t_array: np.ndarray,
) -> float:
    """Average growth rate over time.

    Parameters
    ----------
    h_history : np.ndarray, shape (n_times, n_electrode)
        Thickness snapshots.
    t_array : np.ndarray, shape (n_times,)
        Time values [s].

    Returns
    -------
    float
        Mean dh/dt [m/s].
    """
    if len(t_array) < 2:
        return 0.0
    mean_h = np.mean(h_history, axis=1)
    total_growth = mean_h[-1] - mean_h[0]
    total_time = t_array[-1] - t_array[0]
    if total_time == 0:
        return 0.0
    return float(total_growth / total_time)


def total_charge_proxy(
    flux_history: np.ndarray,
    dt_array: np.ndarray,
) -> float:
    """Integrated reaction flux as charge proxy.

    Q_proxy = sum(flux_i * dt_i)

    Parameters
    ----------
    flux_history : np.ndarray
        Total flux at each timestep [mol/(m*s)].
    dt_array : np.ndarray
        Time step sizes [s].

    Returns
    -------
    float
        Total charge proxy [mol/m].
    """
    n = min(len(flux_history), len(dt_array))
    return float(np.sum(flux_history[:n] * dt_array[:n]))


def roughness_metric(h: np.ndarray, dx: float = 1.0) -> float:
    """Surface roughness as integral of |grad(h)| / length.

    Parameters
    ----------
    h : np.ndarray, shape (n_electrode,)
        Film thickness [m].
    dx : float
        Grid spacing along electrode [m].

    Returns
    -------
    float
        Roughness metric (dimensionless ratio).
    """
    if len(h) < 2:
        return 0.0
    grad_h = np.abs(np.diff(h) / dx)
    L = dx * (len(h) - 1)
    if L == 0:
        return 0.0
    return float(np.sum(grad_h * dx) / L)


def compute_all_metrics(
    h: np.ndarray,
    C_field: np.ndarray,
    C_bulk: float,
    grid: Grid2D,
    electrode_mask: np.ndarray,
    D: float,
    k_rxn: float,
    L_electrode: float,
    flux_history: Optional[np.ndarray] = None,
    dt_array: Optional[np.ndarray] = None,
    h_history: Optional[np.ndarray] = None,
    t_array: Optional[np.ndarray] = None,
) -> dict:
    """Compute all morphology metrics.

    Returns
    -------
    dict
        Keys: AI, UI, depletion_depth, timescale_ratio,
              mean_growth_rate, total_charge_proxy, roughness.
    """
    ai = anisotropy_index(h)
    ui = uniformity_index(h)
    dd = depletion_depth(C_field, C_bulk, grid, electrode_mask)
    tr = timescale_ratio(D, k_rxn, L_electrode)

    mgr = 0.0
    if h_history is not None and t_array is not None:
        mgr = mean_growth_rate(h_history, t_array)

    tcp = 0.0
    if flux_history is not None and dt_array is not None:
        tcp = total_charge_proxy(flux_history, dt_array)

    rough = roughness_metric(h, grid.dx)

    return {
        "AI": ai,
        "UI": ui,
        "depletion_depth": dd,
        "timescale_ratio": tr,
        "mean_growth_rate": mgr,
        "total_charge_proxy": tcp,
        "roughness": rough,
    }
