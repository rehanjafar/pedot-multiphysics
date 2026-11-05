"""
Validation scaffold for comparing simulation with experimental data.

Accepted experimental data formats:
- CV curve: CSV with columns V, I
- CA curve: CSV with columns t, I

Even without experimental data, this module provides the interface
for future validation workflows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils_io import ensure_dir, save_json

logger = logging.getLogger(__name__)


def load_experimental_cv(path: Path) -> pd.DataFrame:
    """Load cyclic voltammetry data.

    Expected format: CSV with columns 'V' (potential) and 'I' (current).

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
        With columns V [V] and I [A].
    """
    df = pd.read_csv(Path(path))
    required = {"V", "I"}
    if not required.issubset(df.columns):
        raise ValueError(f"CV data must have columns {required}, got {set(df.columns)}")
    return df[["V", "I"]]


def load_experimental_ca(path: Path) -> pd.DataFrame:
    """Load chronoamperometry data.

    Expected format: CSV with columns 't' (time) and 'I' (current).

    Parameters
    ----------
    path : Path

    Returns
    -------
    pd.DataFrame
        With columns t [s] and I [A].
    """
    df = pd.read_csv(Path(path))
    required = {"t", "I"}
    if not required.issubset(df.columns):
        raise ValueError(f"CA data must have columns {required}, got {set(df.columns)}")
    return df[["t", "I"]]


def align_and_compare(
    sim_data: pd.DataFrame,
    exp_data: pd.DataFrame,
    x_col: str = "t",
    y_col: str = "I",
) -> dict:
    """Align simulation and experimental data, compute error metrics.

    Parameters
    ----------
    sim_data : pd.DataFrame
    exp_data : pd.DataFrame
    x_col : str
        Independent variable column.
    y_col : str
        Dependent variable column.

    Returns
    -------
    dict
        Keys: RMSE, R_squared, max_error, mean_error.
    """
    # Interpolate simulation to experimental x values
    sim_interp = np.interp(exp_data[x_col].values, sim_data[x_col].values, sim_data[y_col].values)
    exp_vals = exp_data[y_col].values

    residuals = sim_interp - exp_vals
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((exp_vals - np.mean(exp_vals)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "RMSE": rmse,
        "R_squared": r_squared,
        "max_error": float(np.max(np.abs(residuals))),
        "mean_error": float(np.mean(np.abs(residuals))),
    }


def plot_validation_overlay(
    sim_data: pd.DataFrame,
    exp_data: pd.DataFrame,
    save_path: Path,
    x_col: str = "t",
    y_col: str = "I",
    dpi: int = 300,
) -> None:
    """Plot simulation vs experimental overlay."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(exp_data[x_col], exp_data[y_col], "ko", markersize=3, alpha=0.6, label="Experimental")
    ax.plot(sim_data[x_col], sim_data[y_col], "r-", linewidth=2, label="Simulation")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Simulation vs Experimental Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(save_path), dpi=dpi)
    plt.close(fig)


@dataclass
class ValidationReport:
    """Container for validation results."""
    metrics: dict = field(default_factory=dict)
    plots: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def save(self, output_dir: Path) -> None:
        output_dir = ensure_dir(Path(output_dir))
        save_json(self.metrics, output_dir / "validation_metrics.json")
        with open(output_dir / "validation_notes.txt", "w") as f:
            for note in self.notes:
                f.write(note + "\n")


def run_validation(
    sim_timeseries_path: Path,
    exp_data_path: Path,
    output_dir: Path,
    data_type: str = "CA",
) -> ValidationReport:
    """Run validation comparison.

    Parameters
    ----------
    sim_timeseries_path : Path
        Path to simulation timeseries.csv.
    exp_data_path : Path
        Path to experimental data CSV.
    output_dir : Path
    data_type : str
        "CV" or "CA".

    Returns
    -------
    ValidationReport
    """
    output_dir = ensure_dir(Path(output_dir))
    report = ValidationReport()

    try:
        sim_df = pd.read_csv(sim_timeseries_path)

        if data_type == "CV":
            exp_df = load_experimental_cv(exp_data_path)
            x_col, y_col = "V", "I"
        else:
            exp_df = load_experimental_ca(exp_data_path)
            x_col, y_col = "t", "I"

        # Map simulation columns
        if y_col == "I" and "total_flux" in sim_df.columns:
            sim_df["I"] = sim_df["total_flux"]  # Proxy

        metrics = align_and_compare(sim_df, exp_df, x_col, y_col)
        report.metrics = metrics
        report.notes.append(f"Validation type: {data_type}")
        report.notes.append(f"RMSE: {metrics['RMSE']:.6f}")
        report.notes.append(f"R²: {metrics['R_squared']:.4f}")

        plot_path = output_dir / f"validation_{data_type.lower()}.png"
        plot_validation_overlay(sim_df, exp_df, plot_path, x_col, y_col)
        report.plots.append(plot_path)

    except Exception as e:
        report.notes.append(f"Validation failed: {e}")
        logger.error("Validation failed: %s", e)

    report.save(output_dir)
    return report
