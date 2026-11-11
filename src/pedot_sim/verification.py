"""
Numerical verification: mesh and timestep refinement studies.

Demonstrates convergence of AI and UI with grid/timestep refinement.
Uses Richardson extrapolation for error estimation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimConfig
from .runner_single import run_single
from .utils_io import ensure_dir, load_json, save_csv, save_json

logger = logging.getLogger(__name__)


def run_mesh_refinement(
    config: SimConfig,
    campaign_id: str,
    output_base: Path,
    mesh_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """Run mesh refinement study.

    Runs same condition with different Nx=Ny and compares AI/UI convergence.

    Parameters
    ----------
    config : SimConfig
        Base configuration.
    campaign_id : str
    output_base : Path
    mesh_sizes : list[int], optional
        Grid sizes to test. Default: [50, 100, 200].

    Returns
    -------
    pd.DataFrame
        Convergence data.
    """
    if mesh_sizes is None:
        mesh_sizes = [50, 100, 200]

    output_base = Path(output_base)
    results = []

    for N in mesh_sizes:
        cfg = config.copy()
        cfg.numerical.Nx = N
        cfg.numerical.Ny = N
        cfg.nucleation.enable_noise = False
        run_id = f"verify_mesh_N{N}"

        logger.info("Mesh refinement: N=%d", N)
        try:
            metrics = run_single(cfg, run_id, output_base, campaign_id)
            results.append({
                "N": N,
                "dx": (2*cfg.geometry.compartment_width + cfg.geometry.bridge_length) / (N-1),
                "AI": metrics.get("AI", np.nan),
                "UI": metrics.get("UI", np.nan),
                "depletion_depth": metrics.get("depletion_depth", np.nan),
                "mean_h": metrics.get("mean_h", np.nan),
                "effective_dt": metrics.get("effective_dt", np.nan),
            })
        except Exception as e:
            logger.error("Mesh refinement failed for N=%d: %s", N, e)
            results.append({"N": N, "dx": np.nan, "AI": np.nan, "UI": np.nan})

    df = pd.DataFrame(results)

    # Richardson extrapolation (if 3+ points)
    if len(df) >= 3 and not df["AI"].isna().any():
        df["AI_error_estimate"] = _richardson_error(df["AI"].values, df["dx" if "dx" in df else "effective_dt"].values)
        df["UI_error_estimate"] = _richardson_error(df["UI"].values, df["dx" if "dx" in df else "effective_dt"].values)

    # Save
    verify_dir = ensure_dir(output_base / "verification" / campaign_id)
    save_csv(df, verify_dir / "mesh_refinement.csv")

    return df


def run_timestep_refinement(
    config: SimConfig,
    campaign_id: str,
    output_base: Path,
    dt_values: list[float] | None = None,
) -> pd.DataFrame:
    """Run timestep refinement study.

    Parameters
    ----------
    config : SimConfig
    campaign_id : str
    output_base : Path
    dt_values : list[float], optional
        Timesteps to test. Default: [0.01, 0.005, 0.001].

    Returns
    -------
    pd.DataFrame
    """
    if dt_values is None:
        from .waveforms import integration_times
        base_dt = float(np.max(np.diff(integration_times(config))))
        dt_values = [base_dt, base_dt/2, base_dt/4]

    output_base = Path(output_base)
    results = []

    for dt_val in dt_values:
        cfg = config.copy()
        cfg.numerical.dt = dt_val
        cfg.nucleation.enable_noise = False
        run_id = f"verify_dt_{dt_val:.17g}".replace("+", "").replace("-", "m")

        logger.info("Timestep refinement: dt=%e", dt_val)
        try:
            metrics = run_single(cfg, run_id, output_base, campaign_id)
            results.append({
                "dt": dt_val,
                "AI": metrics.get("AI", np.nan),
                "UI": metrics.get("UI", np.nan),
                "depletion_depth": metrics.get("depletion_depth", np.nan),
                "mean_h": metrics.get("mean_h", np.nan),
                "effective_dt": metrics.get("effective_dt", np.nan),
            })
        except Exception as e:
            logger.error("Timestep refinement failed for dt=%e: %s", dt_val, e)
            results.append({"dt": dt_val, "AI": np.nan, "UI": np.nan})

    df = pd.DataFrame(results)

    if len(df) >= 3 and not df["AI"].isna().any():
        df["AI_error_estimate"] = _richardson_error(df["AI"].values, df["dx" if "dx" in df else "effective_dt"].values)
        df["UI_error_estimate"] = _richardson_error(df["UI"].values, df["dx" if "dx" in df else "effective_dt"].values)

    verify_dir = ensure_dir(output_base / "verification" / campaign_id)
    save_csv(df, verify_dir / "timestep_refinement.csv")

    return df


def _richardson_error(values, spacings):
    """Fine-level estimate using observed order; NaN when not identifiable.

    Three monotone values and strictly decreasing actual spacings are required.
    No assumed convergence order or fixed refinement ratio is used.
    """
    from scipy.optimize import brentq
    errors = np.full(len(values), np.nan)
    if len(values) != 3 or not np.isfinite(values).all():
        return errors
    h = np.asarray(spacings) / max(spacings)
    if not (h[0] > h[1] > h[2] > 0):
        return errors
    d0, d1 = values[0]-values[1], values[1]-values[2]
    if d0*d1 <= 0:
        return errors
    def residual(order):
        return (h[0]**order-h[1]**order)/(h[1]**order-h[2]**order) - d0/d1
    try:
        order = brentq(residual, .05, 10.)
        errors[-1] = abs(d1) / ((h[1]/h[2])**order-1)
    except ValueError:
        pass
    return errors


def plot_convergence(
    df: pd.DataFrame,
    x_col: str,
    save_path: Path,
    title: str = "Convergence Study",
    dpi: int = 300,
) -> None:
    """Plot convergence of AI and UI vs refinement parameter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, dv in enumerate(["AI", "UI"]):
        ax = axes[idx]
        if dv in df.columns:
            ax.plot(df[x_col], df[dv], "o-", linewidth=2, markersize=8)
            ax.set_xlabel(x_col)
            ax.set_ylabel(dv)
            ax.set_title(f"{dv} Convergence")
            ax.grid(True, alpha=0.3)

            error_col = f"{dv}_error_estimate"
            if error_col in df.columns:
                valid = ~df[error_col].isna()
                if valid.any():
                    ax.fill_between(
                        df.loc[valid, x_col],
                        df.loc[valid, dv] - df.loc[valid, error_col],
                        df.loc[valid, dv] + df.loc[valid, error_col],
                        alpha=0.2,
                    )

    plt.suptitle(title)
    plt.tight_layout()
    fig.savefig(Path(save_path), dpi=dpi)
    plt.close(fig)


def run_full_verification(
    config: SimConfig,
    campaign_id: str,
    output_base: Path,
) -> None:
    """Run complete verification suite."""
    output_base = Path(output_base)
    fig_dir = ensure_dir(output_base / "reports" / campaign_id / "figures")

    # Verify the requested physical duration; do not silently shorten the experiment.
    cfg = config.copy()
    cfg.nucleation.enable_noise = False

    logger.info("Running mesh refinement study...")
    mesh_df = run_mesh_refinement(cfg, campaign_id, output_base)
    plot_convergence(mesh_df, "N", fig_dir / "verification_mesh.png", "Mesh Refinement")

    logger.info("Running timestep refinement study...")
    dt_df = run_timestep_refinement(cfg, campaign_id, output_base)
    plot_convergence(dt_df, "dt", fig_dir / "verification_timestep.png", "Timestep Refinement")

    # Passing these output tolerances does not establish empirical validity.
    failures = []
    for name, frame in (("mesh", mesh_df), ("timestep", dt_df)):
        if not np.isfinite(frame[["AI", "UI", "mean_h"]].values).all():
            failures.append(f"{name} refinement contains failed/nonfinite runs")
            continue
        if name == "timestep" and frame.effective_dt.nunique() < 3:
            failures.append("Waveform timestep cap collapsed refinement levels")
        for metric, floor in (("AI", 1e-8), ("UI", 1e-12), ("mean_h", 1e-12)):
            coarse, fine = frame[metric].values[-2:]
            if abs(coarse-fine) > max(floor, .05*abs(fine)):
                failures.append(f"{name} refinement did not meet 5% tolerance for {metric}")
    save_json({"passed": not failures, "failures": failures, "t_end": cfg.numerical.t_end,
               "physics_stage": cfg.physics.physics_stage, "relative_tolerance": .05,
               "interpretation": "Numerical output agreement only; not experimental validation."},
              ensure_dir(output_base / "verification" / campaign_id) / "verification_summary.json")
    if failures:
        raise ValueError("; ".join(failures))
    logger.info("Deterministic refinement tolerances passed (not experimental validation)")
