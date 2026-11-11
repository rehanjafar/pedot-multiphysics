"""
2D visualization module for per-run and campaign-level figures.

All plots use consistent styling with proper labels, units, and colorbars.
Matplotlib backend set to 'Agg' for headless operation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimConfig
from .geometry import DomainInfo
from .utils_grid import Grid2D
from .utils_io import ensure_dir

logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def plot_potential_map(
    V: np.ndarray,
    grid: Grid2D,
    save_path: Path,
    domain_mask: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """Plot electric potential V(x,y) with contour lines."""
    fig, ax = plt.subplots(figsize=(8, 5))
    V_plot = V.copy()
    if domain_mask is not None:
        V_plot = np.ma.masked_where(~domain_mask, V_plot)
    im = ax.pcolormesh(grid.X * 1e3, grid.Y * 1e3, V_plot, cmap="RdBu_r", shading="auto")
    cs = ax.contour(grid.X * 1e3, grid.Y * 1e3, V_plot, levels=15, colors="k", linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7)
    plt.colorbar(im, ax=ax, label="Potential V [V]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Electric Potential Distribution")
    ax.set_aspect("equal")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_field_magnitude(
    E_mag: np.ndarray,
    grid: Grid2D,
    save_path: Path,
    domain_mask: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """Plot |E| heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    E_plot = E_mag.copy()
    if domain_mask is not None:
        E_plot = np.ma.masked_where(~domain_mask, E_plot)
    im = ax.pcolormesh(grid.X * 1e3, grid.Y * 1e3, E_plot, cmap="hot", shading="auto")
    plt.colorbar(im, ax=ax, label="|E| [V/m]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Electric Field Magnitude")
    ax.set_aspect("equal")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_concentration(
    C: np.ndarray,
    grid: Grid2D,
    save_path: Path,
    C_bulk: float = 50.0,
    domain_mask: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """Plot EDOT concentration heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    C_plot = C.copy()
    if domain_mask is not None:
        C_plot = np.ma.masked_where(~domain_mask, C_plot)
    im = ax.pcolormesh(grid.X * 1e3, grid.Y * 1e3, C_plot, cmap="viridis", shading="auto",
                       vmin=0, vmax=C_bulk)
    plt.colorbar(im, ax=ax, label="[EDOT] [mol/m³]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("EDOT Monomer Concentration")
    ax.set_aspect("equal")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_depletion_contour(
    C: np.ndarray,
    grid: Grid2D,
    C_bulk: float,
    save_path: Path,
    domain_mask: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """Plot depletion zone contour overlay (C = 0.9 * C_bulk)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    C_plot = C.copy()
    if domain_mask is not None:
        C_plot = np.ma.masked_where(~domain_mask, C_plot)
    im = ax.pcolormesh(grid.X * 1e3, grid.Y * 1e3, C_plot, cmap="viridis", shading="auto",
                       vmin=0, vmax=C_bulk)
    # Depletion contour
    threshold = 0.9 * C_bulk
    C_filled = np.where(domain_mask, C, C_bulk) if domain_mask is not None else C
    ax.contour(grid.X * 1e3, grid.Y * 1e3, C_filled, levels=[threshold],
               colors="red", linewidths=2, linestyles="dashed")
    plt.colorbar(im, ax=ax, label="[EDOT] [mol/m³]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Depletion Zone (red dashed = 0.9·C_bulk)")
    ax.set_aspect("equal")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_reaction_rate(
    R: np.ndarray,
    grid: Grid2D,
    save_path: Path,
    domain_mask: Optional[np.ndarray] = None,
    dpi: int = 300,
) -> None:
    """Plot reaction rate heatmap."""
    fig, ax = plt.subplots(figsize=(8, 5))
    R_plot = R.copy()
    if domain_mask is not None:
        R_plot = np.ma.masked_where(~domain_mask, R_plot)
    im = ax.pcolormesh(grid.X * 1e3, grid.Y * 1e3, R_plot, cmap="magma", shading="auto")
    plt.colorbar(im, ax=ax, label="R [mol/(m³·s)]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title("Reaction Rate Distribution")
    ax.set_aspect("equal")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_thickness_heatmap(
    h: np.ndarray,
    electrode_x: np.ndarray,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Plot film thickness as a 1D bar/profile along electrode."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(electrode_x * 1e3, h * 1e6, color="steelblue", marker=".", linewidth=1.5)
    ax.set_xlabel("x along electrode [mm]")
    ax.set_ylabel("Film thickness [µm]")
    ax.set_title("PEDOT Film Thickness Profile")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_thickness_crosssection(
    h: np.ndarray,
    electrode_x: np.ndarray,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Plot thickness cross-section along electrode centerline."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(electrode_x * 1e3, h * 1e6, "b-", linewidth=2)
    ax.fill_between(electrode_x * 1e3, 0, h * 1e6, alpha=0.3, color="steelblue")
    ax.set_xlabel("x along electrode [mm]")
    ax.set_ylabel("Film thickness [µm]")
    ax.set_title("Film Thickness Cross-Section (Centerline)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_timeseries_thickness(
    t: np.ndarray,
    mean_h: np.ndarray,
    max_h: np.ndarray,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Plot thickness time series."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, np.array(mean_h) * 1e6, "b-", label="Mean h", linewidth=2)
    ax.plot(t, np.array(max_h) * 1e6, "r--", label="Max h", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Film thickness [µm]")
    ax.set_title("Film Thickness Evolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_timeseries_flux(
    t: np.ndarray,
    flux: np.ndarray,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Plot current proxy (integrated flux) time series."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, flux, "g-", linewidth=1.5)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total flux [mol/(m·s)]")
    ax.set_title("Reaction Flux (Current Proxy)")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def plot_timescale_panel(
    tau_diff: float,
    tau_rxn: float,
    ratio: float,
    save_path: Path,
    dpi: int = 300,
) -> None:
    """Summary panel of timescale comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Bar chart
    ax = axes[0]
    values=np.array([tau_diff,tau_rxn],float)
    bars = ax.bar(["τ_diff", "τ_rxn"], np.where(np.isfinite(values),values,0), color=["royalblue", "coral"])
    ax.set_ylabel("Timescale [s]")
    ax.set_title("Competing Timescales")
    for bar, val in zip(bars, [tau_diff, tau_rxn]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2e}",
                ha="center", va="bottom", fontsize=9)
    # Ratio indicator
    ax = axes[1]
    ax.barh(["τ_diff / τ_rxn"], [ratio if np.isfinite(ratio) else 0], color="steelblue")
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Ratio")
    ax.set_title(f"Timescale Ratio = {ratio:.3f}")
    ax.text(.5, -.25, "Characteristic times only; no validated regime classification",
            transform=ax.transAxes, ha="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)


def generate_all_run_figures(
    run_dir: Path,
    config: SimConfig,
    grid: Grid2D,
    domain: DomainInfo,
    C_final: np.ndarray,
    V_final: Optional[np.ndarray],
    E_final: Optional[np.ndarray],
    R_final: Optional[np.ndarray],
    h_final: np.ndarray,
    electrode_x: np.ndarray,
    ts_df: pd.DataFrame,
    dpi: int = 300,
) -> None:
    """Generate all per-run 2D figures."""
    fig_dir = ensure_dir(Path(run_dir) / "figures")
    dm = domain.domain_mask
    C_bulk = config.physics.C_bulk

    logger.info("Generating 2D figures in %s", fig_dir)

    # Potential map
    if V_final is not None:
        plot_potential_map(V_final, grid, fig_dir / "potential_map.png", dm, dpi)
        if E_final is not None:
            plot_field_magnitude(E_final, grid, fig_dir / "field_magnitude.png", dm, dpi)

    # Concentration
    plot_concentration(C_final, grid, fig_dir / "concentration.png", C_bulk, dm, dpi)
    plot_depletion_contour(C_final, grid, C_bulk, fig_dir / "depletion_zone.png", dm, dpi)

    # Reaction rate
    if R_final is not None:
        plot_reaction_rate(R_final, grid, fig_dir / "reaction_rate.png", dm, dpi)

    # Thickness
    if len(electrode_x) > 0 and len(h_final) > 0:
        plot_thickness_heatmap(h_final, electrode_x, fig_dir / "thickness_heatmap.png", dpi)
        plot_thickness_crosssection(h_final, electrode_x, fig_dir / "thickness_crosssection.png", dpi)

    # Time series
    if not ts_df.empty:
        t = ts_df["time"].values
        plot_timeseries_thickness(t, ts_df["mean_h"].values, ts_df["max_h"].values,
                                  fig_dir / "timeseries_thickness.png", dpi)
        plot_timeseries_flux(t, ts_df["flux"].values, fig_dir / "timeseries_flux.png", dpi)

    # Timescale panel
    D = config.physics.diffusion_coefficient
    k = config.physics.k_rxn
    L = config.geometry.electrode_length
    tau_d = L ** 2 / D if D > 0 else float("inf")
    tau_r = L / k if k > 0 else float("inf")
    ratio = tau_d / tau_r if tau_r > 0 else float("inf")
    plot_timescale_panel(tau_d, tau_r, ratio, fig_dir / "timescale_panel.png", dpi)

    logger.info("2D figures generated successfully")
