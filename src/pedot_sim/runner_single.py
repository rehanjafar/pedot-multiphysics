"""
Single-run executor for PEDOT electropolymerization simulation.

Orchestrates a complete simulation run: geometry construction, field
initialisation, time-stepping via the coupled physics in ``growth.py``,
post-processing diagnostics via ``metrics.py``, and output serialisation
to the canonical directory layout.
"""

from __future__ import annotations

import logging
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pedot_sim.config import SimConfig
from pedot_sim.geometry import DomainInfo, build_domain, get_electrode_x
from pedot_sim.growth import StepResult, evolve_one_timestep
from pedot_sim.metrics import (
    anisotropy_index,
    compute_all_metrics,
    depletion_depth,
    timescale_ratio,
    uniformity_index,
)
from pedot_sim.nucleation import NucleationModel
from pedot_sim.utils_io import (
    ensure_dir,
    save_json,
    save_npz,
    save_csv,
    runtime_provenance,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight per-timestep record
# ---------------------------------------------------------------------------

@dataclass
class TimeseriesRecord:
    """One row of the per-timestep timeseries output."""

    time: float
    mean_h: float
    max_h: float
    asymmetry_index: float
    uniformity_index: float
    flux: float
    depletion_depth: float


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_fields_npz(
    snapshots: List[Dict[str, Any]],
    path: Path,
) -> None:
    """Save concentration + thickness snapshots as compressed ``.npz``."""
    arrays: Dict[str, np.ndarray] = {}
    times: List[float] = []
    for i, snap in enumerate(snapshots):
        arrays[f"C_{i:04d}"] = snap["C"]
        arrays[f"h_{i:04d}"] = snap["h"]
        times.append(snap["time"])
    arrays["snapshot_times"] = np.array(times)
    save_npz(arrays, path)


def _save_thickness_npz(
    snapshots: List[Dict[str, Any]],
    electrode_x: np.ndarray,
    path: Path,
) -> None:
    """Save thickness profile snapshots as compressed ``.npz``."""
    arrays: Dict[str, np.ndarray] = {"x": electrode_x}
    times: List[float] = []
    for i, snap in enumerate(snapshots):
        arrays[f"h_{i:04d}"] = snap["h"]
        times.append(snap["time"])
    arrays["snapshot_times"] = np.array(times)
    save_npz(arrays, path)


def _save_timeseries_csv(
    records: List[TimeseriesRecord],
    path: Path,
) -> None:
    """Write per-timestep diagnostics to CSV."""
    rows = [
        {
            "time": r.time,
            "mean_h": r.mean_h,
            "max_h": r.max_h,
            "asymmetry_index": r.asymmetry_index,
            "uniformity_index": r.uniformity_index,
            "flux": r.flux,
            "depletion_depth": r.depletion_depth,
        }
        for r in records
    ]
    df = pd.DataFrame(rows)
    save_csv(df, path)


# ---------------------------------------------------------------------------
# Figure generation (best-effort; failures logged, not fatal)
# ---------------------------------------------------------------------------

def _try_generate_figures(
    run_dir: Path,
    config: SimConfig,
    domain: DomainInfo,
    snapshots: List[Dict[str, Any]],
    electrode_x: np.ndarray,
    timeseries: List[TimeseriesRecord],
) -> None:
    """Best-effort figure generation.  Failures are logged but not fatal."""
    fig_dir = ensure_dir(run_dir / "figures")

    try:
        _plot_thickness_evolution(snapshots, electrode_x, fig_dir, config)
    except Exception:
        logger.warning("Failed to generate thickness evolution figure", exc_info=True)

    try:
        _plot_timeseries(timeseries, fig_dir, config)
    except Exception:
        logger.warning("Failed to generate timeseries figure", exc_info=True)

    try:
        _plot_concentration_snapshots(snapshots, domain, config, fig_dir)
    except Exception:
        logger.warning("Failed to generate concentration snapshot figure", exc_info=True)


def _plot_thickness_evolution(
    snapshots: List[Dict[str, Any]],
    electrode_x: np.ndarray,
    out_dir: Path,
    config: SimConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for snap in snapshots:
        ax.plot(
            electrode_x * 1e3,
            snap["h"] * 1e6,
            label=f"t = {snap['time']:.1f} s",
        )
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("Film thickness [um]")
    ax.set_title(
        f"PEDOT thickness evolution  (f={config.waveform.frequency:.0f} Hz, "
        f"D={config.physics.diffusion_coefficient:.1e} m\u00b2/s)"
    )
    if len(snapshots) <= 10:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / f"thickness_evolution.{config.output.figure_format}", dpi=config.output.dpi)
    plt.close(fig)


def _plot_timeseries(
    records: List[TimeseriesRecord],
    out_dir: Path,
    config: SimConfig,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = [r.time for r in records]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(times, [r.mean_h * 1e6 for r in records])
    axes[0].set_ylabel("Mean h [\u03bcm]")

    axes[1].plot(times, [r.asymmetry_index for r in records])
    axes[1].set_ylabel("Peak/mean thickness (AI)")

    axes[2].plot(times, [r.depletion_depth * 1e3 for r in records])
    axes[2].set_ylabel("Depletion depth [mm]")
    axes[2].set_xlabel("Time [s]")

    fig.suptitle("Timeseries diagnostics")
    fig.tight_layout()
    fig.savefig(out_dir / f"timeseries.{config.output.figure_format}", dpi=config.output.dpi)
    plt.close(fig)


def _plot_concentration_snapshots(
    snapshots: List[Dict[str, Any]],
    domain: DomainInfo,
    config: SimConfig,
    out_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(snapshots), 6)
    if n == 0:
        return
    indices = np.linspace(0, len(snapshots) - 1, n, dtype=int)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), layout="constrained")
    if n == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        snap = snapshots[idx]
        C_plot = snap["C"].copy()
        C_plot[~domain.domain_mask] = np.nan  # mask outside domain
        im = ax.imshow(
            C_plot,
            extent=[0, domain.Lx * 1e3, 0, domain.Ly * 1e3],
            origin="lower",
            aspect="auto",
            vmin=0,
            vmax=config.physics.C_bulk,
        )
        ax.set_title(f"t = {snap['time']:.1f} s")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")

    fig.colorbar(im, ax=axes, label="C [mol/m\u00b3]", shrink=0.8)
    fig.savefig(out_dir / f"concentration_snapshots.{config.output.figure_format}", dpi=config.output.dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_single(
    config: SimConfig,
    run_id: str,
    output_base: Path,
    campaign_id: str = "default",
) -> Dict[str, Any]:
    """Execute a single PEDOT electropolymerization simulation.

    Parameters
    ----------
    config : SimConfig
        Full simulation configuration (all sub-configs populated).
    run_id : str
        Unique identifier for this run.
    output_base : Path
        Root data directory (should contain ``simulated_raw/`` etc.).
    campaign_id : str
        Campaign grouping identifier.

    Returns
    -------
    dict
        Metrics dictionary summarising the run outcome.  On failure the
        dict will contain ``"status": "failed"`` and an ``"error"`` key.
    """
    from .research import process_diagnostics, ohmic_power_per_depth
    from .waveforms import integration_times
    from .config import validate_config
    from .utils_io import validate_id, safe_output_path
    validate_config(config)
    validate_id(campaign_id); validate_id(run_id)
    output_base = Path(output_base)
    # ---- Output directory tree -------------------------------------------
    run_dir = safe_output_path(output_base, "simulated_raw", campaign_id, run_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)  # Atomic claim: concurrent runs cannot overwrite each other.
    for sub in ("figures", "animations", "logs"):
        ensure_dir(run_dir / sub)

    # ---- Per-run file logger ---------------------------------------------
    log_path = run_dir / "logs" / "run.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-7s] %(name)s: %(message)s")
    )
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    logger.info("=== Starting run %s (campaign=%s) ===", run_id, campaign_id)
    start_iso = datetime.now(timezone.utc).isoformat()
    wall_t0 = time.perf_counter()

    # Save params early so a partial/failed run is still reproducible
    save_json(
        {
            "geometry": {k: getattr(config.geometry, k) for k in config.geometry.__dataclass_fields__},
            "waveform": {k: getattr(config.waveform, k) for k in config.waveform.__dataclass_fields__},
            "physics": {k: getattr(config.physics, k) for k in config.physics.__dataclass_fields__},
            "numerical": {k: getattr(config.numerical, k) for k in config.numerical.__dataclass_fields__},
            "nucleation": {k: getattr(config.nucleation, k) for k in config.nucleation.__dataclass_fields__},
            "output": {k: getattr(config.output, k) for k in config.output.__dataclass_fields__},
        },
        run_dir / "params.json",
    )

    try:
        # ---- Build geometry & domain -------------------------------------
        domain = build_domain(config.geometry, config.numerical)
        grid = domain.grid
        electrode_x = get_electrode_x(domain)

        logger.info(
            "Domain: Lx=%.4f m, Ly=%.4f m, Nx=%d, Ny=%d",
            domain.Lx, domain.Ly, grid.Nx, grid.Ny,
        )

        # ---- Initialise fields -------------------------------------------
        C = np.where(domain.domain_mask, config.physics.C_bulk, 0.)
        initial_C = C.copy()

        # Electrode cells for h
        e_rows, e_cols = np.where(domain.electrode_mask)
        n_electrode = len(e_cols)

        if n_electrode < 2:
            raise ValueError("Resolve the electrode with at least two cells")

        # Nucleation model handles initial perturbation and ongoing noise
        nuc_model = NucleationModel(config.nucleation, n_electrode)
        h = nuc_model.generate_initial_nuclei()

        endpoints = integration_times(config)
        dt = float(np.max(np.diff(endpoints)))
        t_end = config.numerical.t_end
        n_steps = len(endpoints)-1

        # Determine snapshot timestep indices
        n_snapshots = config.numerical.n_snapshots
        snapshot_indices: set = set()
        if n_snapshots > 0 and n_steps > 0:
            snapshot_indices = set(
                np.linspace(0, n_steps - 1, n_snapshots, dtype=int).tolist()
            )
            snapshot_indices.add(n_steps-1)

        # How often to record timeseries (aim for ~1000 rows max)
        ts_stride = max(1, n_steps // 1000)

        logger.info(
            "Time-stepping: n_steps=%d, dt=%.2e s, t_end=%.2f s, snapshots=%d",
            n_steps, dt, t_end, n_snapshots,
        )

        # ---- Time-stepping loop ------------------------------------------
        timeseries: List[TimeseriesRecord] = [TimeseriesRecord(0., 0., 0., 1., 0., 0., 0.)]
        snapshots: List[Dict[str, Any]] = [{"time": 0., "C": C.copy(), "h": h.copy()}] if n_snapshots else []
        initial_h = h.copy()
        flux_history: List[float] = []
        V_prev: Optional[np.ndarray] = None
        consumed_total = reservoir_total = 0.
        energy = None if config.physics.physics_stage == "A" else 0.
        max_laplace_residual = 0.

        for step in range(n_steps):
            t = float(endpoints[step])
            step_dt = float(endpoints[step+1]-t)

            # Use the master physics orchestrator from growth.py
            step_result: StepResult = evolve_one_timestep(
                C=C,
                h=h,
                t=t,
                dt=step_dt,
                config=config,
                domain=domain,
                nucleation_model=nuc_model,
                V_prev=V_prev,
            )
            C = step_result.C
            h = step_result.h
            V_prev = step_result.V_field
            flux_history.append(step_result.total_flux * step_dt)
            t = float(endpoints[step+1])
            consumed_total += step_result.solver_info["consumed_mol_per_m"]
            reservoir_total += step_result.solver_info["reservoir_exchange_mol_per_m"]
            max_laplace_residual = max(max_laplace_residual, step_result.solver_info.get("laplace_residual", 0.))
            if energy is not None:
                energy += ohmic_power_per_depth(V_prev, domain, config.physics.conductivity)*step_dt

            # Record timeseries
            record_ts = (step % ts_stride == 0) or (step in snapshot_indices) or step == n_steps-1
            if record_ts and n_electrode > 0:
                mean_h_val = float(np.mean(h))
                max_h_val = float(np.max(h))
                ai = anisotropy_index(h)
                ui = uniformity_index(h)
                dd = depletion_depth(
                    C, config.physics.C_bulk, grid, domain.electrode_mask,
                    domain_mask=domain.domain_mask
                )
                timeseries.append(TimeseriesRecord(
                    time=t,
                    mean_h=mean_h_val,
                    max_h=max_h_val,
                    asymmetry_index=ai,
                    uniformity_index=ui,
                    flux=step_result.total_flux,
                    depletion_depth=dd,
                ))

            # Save field snapshot
            if step in snapshot_indices:
                snapshots.append({
                    "time": t,
                    "C": C.copy(),
                    "h": h.copy(),
                })

        # ---- Final diagnostics (via metrics module) ----------------------
        h_history_arr = np.array([snap["h"] for snap in snapshots]) if snapshots else None
        t_snap_arr = np.array([snap["time"] for snap in snapshots]) if snapshots else None
        flux_arr = np.array(flux_history)
        dt_arr = np.ones(len(flux_history))  # history already contains integrated flux

        all_metrics = compute_all_metrics(
            h=h,
            C_field=C,
            C_bulk=config.physics.C_bulk,
            grid=grid,
            electrode_mask=domain.electrode_mask,
            D=config.physics.diffusion_coefficient,
            k_rxn=config.physics.k_rxn,
            L_electrode=config.geometry.electrode_length,
            flux_history=flux_arr,
            dt_array=dt_arr,
            h_history=h_history_arr,
            t_array=t_snap_arr,
            domain_mask=domain.domain_mask,
        )

        final_mean_h = float(np.mean(h)) if n_electrode > 0 else 0.0
        final_max_h = float(np.max(h)) if n_electrode > 0 else 0.0

        # Also compute tau_diff / tau_pulse for the user-facing metrics
        D_val = config.physics.diffusion_coefficient
        L_char = config.geometry.bridge_width / 2.0
        tau_diff = L_char ** 2 / D_val if D_val > 0 else float("inf")
        freq = config.waveform.frequency
        tau_pulse = 1.0 / freq if freq > 0 else float("inf")
        tsr = tau_diff / tau_pulse if tau_pulse > 0 else float("inf")

        metrics: Dict[str, Any] = {
            "run_id": run_id,
            "campaign_id": campaign_id,
            "frequency": config.waveform.frequency,
            "D": config.physics.diffusion_coefficient,
            "bridge_width": config.geometry.bridge_width,
            "geometry": config.geometry.geometry_type,
            "seed": config.nucleation.seed,
            "noise_enabled": config.nucleation.enable_noise,
            "V_DC": config.waveform.V_DC,
            "V_AC": config.waveform.V_AC,
            "waveform_type": config.waveform.waveform_type,
            "duty_cycle": config.waveform.duty_cycle,
            "rest_fraction": config.waveform.rest_fraction,
            "adc_ratio": config.waveform.adc_ratio,
            "t_end": t_end,
            "mean_h": final_mean_h,
            "max_h": final_max_h,
            "AI": all_metrics["AI"],
            "UI": all_metrics["UI"],
            "depletion_depth": all_metrics["depletion_depth"],
            "timescale_ratio": all_metrics["timescale_ratio"],
            "diffusion_pulse_ratio": tsr,
            "physics_stage": config.physics.physics_stage,
            "effective_dt": dt,
            "model_assumption": "weak-coupling fixed-domain approximation",
            "mean_growth_rate": float(np.mean(h-initial_h)) / t_end,
            "total_charge_proxy": all_metrics["total_charge_proxy"],
            "roughness": all_metrics["roughness"],
        }

        process_metrics, audit = process_diagnostics(
            config, domain, initial_C, C, h, consumed_total, reservoir_total, energy)
        metrics.update(process_metrics)
        save_json(audit, run_dir / "research_audit.json")
        # Final state is always saved even when snapshot output is disabled.
        save_npz({"h": h, "x": electrode_x, "C": C,
                  "domain_mask": domain.domain_mask, "electrode_mask": domain.electrode_mask},
                 run_dir / "final_state.npz")

        # ---- Persist outputs ---------------------------------------------
        if config.output.save_fields:
            _save_fields_npz(snapshots, run_dir / "fields.npz")
        _save_thickness_npz(snapshots, electrode_x, run_dir / "thickness.npz")
        _save_timeseries_csv(timeseries, run_dir / "timeseries.csv")
        save_json(metrics, run_dir / "metrics.json")

        # ---- Figures (best-effort) ---------------------------------------
        _try_generate_figures(
            run_dir, config, domain, snapshots, electrode_x, timeseries
        )

        if config.output.save_animations and snapshots:
            from .viz3d import generate_3d_outputs
            generate_3d_outputs(run_dir, np.array([v["h"] for v in snapshots]), h, electrode_x, grid, config.output.animation_fps)

        # ---- Metadata ----------------------------------------------------
        wall_time = time.perf_counter() - wall_t0
        save_json(
            {
                "run_id": run_id,
                "campaign_id": campaign_id,
                "status": "completed",
                "provenance": runtime_provenance(),
                "start_time": start_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "wall_clock_s": wall_time,
                "n_timesteps": n_steps,
                "effective_dt": dt,
                "minimum_dt": float(np.min(np.diff(endpoints))),
                "max_laplace_residual": max_laplace_residual,
                "final_time": t,
                "n_electrode_cells": n_electrode,
                "hostname": platform.node(),
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "error": "",
            },
            run_dir / "metadata.json",
        )

        logger.info(
            "Run %s completed in %.1f s  (mean_h=%.3e m, AI=%.4f, UI=%.4e)",
            run_id, wall_time, final_mean_h,
            all_metrics["AI"], all_metrics["UI"],
        )

        return metrics

    except Exception as exc:
        # ---- Save partial results on failure -----------------------------
        wall_time = time.perf_counter() - wall_t0

        save_json(
            {
                "run_id": run_id,
                "campaign_id": campaign_id,
                "status": "failed",
                "start_time": start_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "wall_clock_s": wall_time,
                "n_timesteps": 0,
                "n_electrode_cells": 0,
                "hostname": platform.node(),
                "python_version": sys.version,
                "numpy_version": np.__version__,
                "error": f"{type(exc).__name__}: {exc}",
            },
            run_dir / "metadata.json",
        )

        # Save whatever timeseries we have
        try:
            if timeseries:  # noqa: F821 - may not be bound
                _save_timeseries_csv(timeseries, run_dir / "timeseries.csv")
        except (NameError, Exception):
            pass

        logger.error("Run %s FAILED after %.1f s: %s", run_id, wall_time, exc)
        logger.debug(traceback.format_exc())

        return {
            "run_id": run_id,
            "campaign_id": campaign_id,
            "status": "failed",
            "error": str(exc),
        }

    finally:
        logger.removeHandler(file_handler)
        file_handler.close()
