"""
Batch campaign executor for PEDOT electropolymerization simulations.

Provides factorial and dense parameter sweeps with parallel execution,
resumability via RunManifest, and result aggregation into a single CSV.
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pedot_sim.config import (
    BatchConfig,
    GeometryConfig,
    NumericalConfig,
    NucleationConfig,
    OutputConfig,
    PhysicsConfig,
    SimConfig,
    WaveformConfig,
)
from pedot_sim.constants import D_LEVELS as _CONST_D_LEVELS
from pedot_sim.utils_io import (
    RunManifest,
    ensure_dir,
    get_campaign_dir,
    load_json,
    save_csv,
    save_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical factorial levels
# ---------------------------------------------------------------------------

_FREQUENCIES: List[float] = [1.0, 10.0, 100.0]

_D_LEVELS: Dict[str, float] = {
    "low": _CONST_D_LEVELS["low"],    # 1e-10  m^2/s  (high viscosity)
    "mid": _CONST_D_LEVELS["mid"],    # 6.5e-10
    "high": _CONST_D_LEVELS["high"],  # 2e-9   (low viscosity)
}

_GEOMETRIES: Dict[str, float] = {
    "narrow": 0.002,
    "wide": 0.005,
}

_N_REPLICATES: int = 3


# ---------------------------------------------------------------------------
# Helper: create a SimConfig copy with overridden parameters
# ---------------------------------------------------------------------------

def _override_config(
    base: SimConfig,
    *,
    frequency: Optional[float] = None,
    D: Optional[float] = None,
    bridge_width: Optional[float] = None,
    geometry_type: Optional[str] = None,
    seed: Optional[int] = None,
) -> SimConfig:
    """Return a deep copy of *base* with the specified fields overridden."""
    cfg = base.copy()
    if frequency is not None:
        cfg.waveform.frequency = frequency
    if D is not None:
        cfg.physics.diffusion_coefficient = D
    if bridge_width is not None:
        cfg.geometry.bridge_width = bridge_width
    if geometry_type is not None:
        cfg.geometry.geometry_type = geometry_type
    if seed is not None:
        cfg.nucleation.seed = seed
    return cfg


# ---------------------------------------------------------------------------
# Factorial run generation
# ---------------------------------------------------------------------------

def generate_factorial_runs(
    config: SimConfig,
    frequencies: Optional[List[float]] = None,
    d_levels: Optional[Dict[str, float]] = None,
    geometries: Optional[Dict[str, float]] = None,
    n_replicates: Optional[int] = None,
    base_seed: Optional[int] = None,
) -> List[Tuple[str, SimConfig]]:
    """Generate a full-factorial parameter sweep.

    Default design: 3 frequencies x 3 D levels x 2 geometries x 3 replicates
    = 54 runs.

    Parameters
    ----------
    config : SimConfig
        Base configuration; swept parameters are overridden per run.
    frequencies : list of float, optional
        Override default frequency list.
    d_levels : dict mapping label -> D value, optional
    geometries : dict mapping label -> bridge_width, optional
    n_replicates : int, optional
    base_seed : int
        Each replicate gets ``seed = base_seed + replicate_index``.

    Returns
    -------
    list of (run_id, SimConfig)
    """
    if frequencies is None:
        frequencies = list(config.batch.frequencies) if config.batch.frequencies else list(_FREQUENCIES)
    if d_levels is None:
        d_levels = {k: _D_LEVELS[k] for k in config.batch.D_levels}
    if geometries is None:
        geometries = {k: _GEOMETRIES[k] for k in config.batch.geometries}
    if n_replicates is None:
        n_replicates = config.batch.n_replicates

    if base_seed is None:
        base_seed = config.nucleation.seed
    runs: List[Tuple[str, SimConfig]] = []

    for freq, (d_label, d_val), (g_label, bw), rep in product(
        frequencies,
        d_levels.items(),
        geometries.items(),
        range(n_replicates),
    ):
        run_id = f"f{freq:.17g}_D{d_label}_g{g_label}_r{rep}"
        run_cfg = _override_config(
            config,
            frequency=freq,
            D=d_val,
            bridge_width=bw,
            geometry_type=g_label,
            seed=base_seed + len(runs),
        )
        runs.append((run_id, run_cfg))

    logger.info("Generated %d factorial runs", len(runs))
    return runs


# ---------------------------------------------------------------------------
# Dense run generation (grid / random / latin hypercube)
# ---------------------------------------------------------------------------

def generate_dense_runs(
    config: SimConfig,
    n_runs: int = 500,
    sampling: str = "latin_hypercube",
    seed: Optional[int] = None,
    freq_range: Optional[Tuple[float, float]] = None,
    D_range: Optional[Tuple[float, float]] = None,
    bw_range: Optional[Tuple[float, float]] = None,
) -> List[Tuple[str, SimConfig]]:
    """Generate a dense parameter sweep for surrogate / ML training.

    Parameters
    ----------
    config : SimConfig
        Base configuration.
    n_runs : int
        Number of sample points (default 500).
    sampling : str
        ``"grid"``, ``"random"``, or ``"latin_hypercube"``.
    seed : int
        RNG seed for reproducibility.
    freq_range, D_range, bw_range : tuple of (low, high), optional
        Physical ranges. Defaults pulled from ``config.batch``.

    Returns
    -------
    list of (run_id, SimConfig)
    """
    if type(n_runs) is not int or n_runs < 1:
        raise ValueError("n_runs must be a positive integer")
    seed = config.nucleation.seed if seed is None else seed
    bc = config.batch
    freq_range = freq_range or tuple(bc.freq_range)
    D_range = D_range or tuple(bc.D_range)
    bw_range = bw_range or (0.001, min(.008, config.geometry.compartment_height))
    ranges = [freq_range, D_range, bw_range, bc.V_DC_range, bc.V_AC_range]
    for low, high in ranges:
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError("Dense ranges must have finite increasing bounds")
    if min(freq_range[0], D_range[0], bw_range[0]) <= 0:
        raise ValueError("Frequency, D and width must be positive")
    rng = np.random.default_rng(seed)
    if sampling == "grid":
        # Three-dimensional grid; waveform amplitudes stay at their configured values.
        count = max(2, int(round(n_runs**(1/3))))
        points = np.array(list(product(np.linspace(*freq_range, count),
            np.geomspace(*D_range, count), np.linspace(*bw_range, count))))
        points = np.column_stack([points, np.full(len(points), config.waveform.V_DC),
                                  np.full(len(points), config.waveform.V_AC)])
    else:
        if sampling in ("latin_hypercube", "lhs"):
            from scipy.stats import qmc
            unit = qmc.LatinHypercube(d=5, seed=seed).random(n_runs)
        elif sampling == "random":
            unit = rng.random((n_runs, 5))
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling}")
        low = np.array([r[0] for r in ranges]); high = np.array([r[1] for r in ranges])
        low[1], high[1] = np.log10(low[1]), np.log10(high[1])
        points = low+unit*(high-low)
        points[:, 1] = 10**points[:, 1]
    runs = []
    for i, (freq, diffusion, width, dc, ac) in enumerate(points):
        cfg = _override_config(config, frequency=float(freq), D=float(diffusion),
                               bridge_width=float(width), geometry_type="custom", seed=seed+i)
        cfg.waveform.V_DC, cfg.waveform.V_AC = float(dc), float(ac)
        runs.append((f"dense_{i:04d}", cfg))
    return runs


def _latin_hypercube_3d(
    n: int,
    freq_range: Tuple[float, float],
    D_range: Tuple[float, float],
    bw_range: Tuple[float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Latin Hypercube Sampling in 3 dimensions.

    ``D`` is sampled in log-space; frequency and bridge-width in linear space.
    """
    samples = np.zeros((n, 3))
    for dim in range(3):
        perm = rng.permutation(n)
        samples[:, dim] = (perm + rng.uniform(size=n)) / n

    points = np.zeros_like(samples)
    # Frequency: linear
    points[:, 0] = freq_range[0] + samples[:, 0] * (freq_range[1] - freq_range[0])
    # D: log-space
    log_lo, log_hi = np.log10(D_range[0]), np.log10(D_range[1])
    points[:, 1] = 10 ** (log_lo + samples[:, 1] * (log_hi - log_lo))
    # Bridge width: linear
    points[:, 2] = bw_range[0] + samples[:, 2] * (bw_range[1] - bw_range[0])

    return points


# ---------------------------------------------------------------------------
# Worker wrapper (must be module-level and picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _run_worker(
    run_id: str,
    config_path_str: str,
    output_base_str: str,
    campaign_id: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """Top-level function executed in a worker process.

    We pass the config path + overrides rather than the full SimConfig to
    avoid pickle issues with nested dataclasses across process boundaries.
    """
    from pedot_sim.config import load_config as _load
    from pedot_sim.runner_single import run_single

    base_cfg = _load(Path(config_path_str))
    cfg = _override_config(
        base_cfg,
        frequency=overrides.get("frequency"),
        D=overrides.get("D"),
        bridge_width=overrides.get("bridge_width"),
        geometry_type=overrides.get("geometry_type"),
        seed=overrides.get("seed"),
    )
    return run_single(cfg, run_id, Path(output_base_str), campaign_id)


def _run_worker_direct(
    run_id: str,
    config_dict: Dict[str, Any],
    output_base_str: str,
    campaign_id: str,
) -> Dict[str, Any]:
    """Alternative worker that receives a serialised config dict.

    Used when no config file path is available (e.g. programmatic usage).
    """
    from pedot_sim.runner_single import run_single

    # Reconstruct SimConfig from dict
    cfg = SimConfig(
        geometry=GeometryConfig(**config_dict.get("geometry", {})),
        waveform=WaveformConfig(**config_dict.get("waveform", {})),
        physics=PhysicsConfig(**config_dict.get("physics", {})),
        numerical=NumericalConfig(**config_dict.get("numerical", {})),
        nucleation=NucleationConfig(**config_dict.get("nucleation", {})),
        output=OutputConfig(**config_dict.get("output", {})),
        batch=BatchConfig(**config_dict.get("batch", {})),
    )
    return run_single(cfg, run_id, Path(output_base_str), campaign_id)


def _config_to_dict(cfg: SimConfig) -> Dict[str, Any]:
    """Serialise a SimConfig to a plain dict suitable for cross-process transfer."""
    return {
        "geometry": {k: getattr(cfg.geometry, k) for k in cfg.geometry.__dataclass_fields__},
        "waveform": {k: getattr(cfg.waveform, k) for k in cfg.waveform.__dataclass_fields__},
        "physics": {k: getattr(cfg.physics, k) for k in cfg.physics.__dataclass_fields__},
        "numerical": {k: getattr(cfg.numerical, k) for k in cfg.numerical.__dataclass_fields__},
        "nucleation": {k: getattr(cfg.nucleation, k) for k in cfg.nucleation.__dataclass_fields__},
        "output": {k: getattr(cfg.output, k) for k in cfg.output.__dataclass_fields__},
        "batch": {k: getattr(cfg.batch, k) for k in cfg.batch.__dataclass_fields__},
    }


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    config: SimConfig,
    campaign_id: str,
    output_base: Optional[Path] = None,
    max_workers: Optional[int] = None,
    sweep: str = "factorial",
    n_dense: Optional[int] = None,
    dense_sampling: Optional[str] = None,
) -> pd.DataFrame:
    """Execute a batch campaign of simulation runs in parallel.

    Parameters
    ----------
    config : SimConfig
        Base configuration.
    campaign_id : str
        Name for this campaign.
    output_base : Path, optional
        Root data directory.  Defaults to ``"data"``.
    max_workers : int
        Number of parallel worker processes.
    sweep : str
        ``"factorial"`` or ``"dense"``.
    n_dense : int
        Number of runs for dense sweep.
    dense_sampling : str
        Sampling strategy for dense sweep.

    Returns
    -------
    pd.DataFrame
        Aggregated results for all completed runs.
    """
    from .config import validate_config
    from .utils_io import validate_id
    validate_config(config); validate_id(campaign_id)
    max_workers = config.batch.max_workers if max_workers is None else max_workers
    if type(max_workers) is not int or max_workers < 1:
        raise ValueError("max_workers must be positive")
    n_dense = config.batch.n_dense_samples if n_dense is None else n_dense
    dense_sampling = dense_sampling or {"lhs": "latin_hypercube", "dense": "latin_hypercube", "factorial": "latin_hypercube"}.get(config.batch.sweep_type, config.batch.sweep_type)
    if n_dense < 1:
        raise ValueError("n_dense must be positive")
    if output_base is None:
        output_base = Path("data")
    output_base = Path(output_base)

    # ---- Generate run list -----------------------------------------------
    if sweep == "factorial":
        runs = generate_factorial_runs(config)
    elif sweep == "dense":
        runs = generate_dense_runs(
            config, n_runs=n_dense, sampling=dense_sampling
        )
    else:
        raise ValueError(f"Unknown sweep type: {sweep!r}")

    if len({rid for rid, _ in runs}) != len(runs):
        raise ValueError("Duplicate run IDs")
    # ---- Check manifest for resumability ---------------------------------
    campaign_dir = get_campaign_dir(output_base, campaign_id)
    # Source identity includes local edits, not only HEAD.
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    provenance = {"config": asdict(config), "source_sha256": digest.hexdigest(),
                  "sweep": sweep, "n_dense": n_dense, "sampling": dense_sampling}
    provenance_path = campaign_dir / "provenance.json"
    if provenance_path.exists() and load_json(provenance_path) != provenance:
        raise ValueError("Campaign configuration/source changed; choose a new campaign ID")
    raw_dir = output_base / "simulated_raw" / campaign_id
    planned = {rid for rid, _ in runs}
    if raw_dir.exists():
        unplanned = {d.name for d in raw_dir.iterdir() if d.is_dir()} - planned
        if unplanned or not provenance_path.exists():
            raise ValueError("Existing campaign outputs do not match this design/provenance; choose a new ID")
    save_json(provenance, provenance_path)
    manifest = RunManifest(campaign_dir)
    for rid in manifest.completed:
        directory = output_base / "simulated_raw" / campaign_id / rid
        if not (directory / "metrics.json").exists() or not (directory / "metadata.json").exists():
            raise ValueError(f"Completed run has missing outputs: {rid}")
        if load_json(directory / "metadata.json").get("status") != "completed":
            raise ValueError(f"Completed run has invalid status: {rid}")

    pending = [(rid, cfg) for rid, cfg in runs if not manifest.is_completed(rid)]
    n_skipped = len(runs) - len(pending)

    if n_skipped > 0:
        logger.info(
            "Resuming campaign %s: %d/%d already completed, %d pending",
            campaign_id, n_skipped, len(runs), len(pending),
        )
    else:
        logger.info(
            "Starting campaign %s: %d runs, %d workers",
            campaign_id, len(pending), max_workers,
        )

    # ---- Execute in parallel ---------------------------------------------
    results: List[Dict[str, Any]] = []
    failed_runs: List[Dict[str, str]] = []
    t0 = time.perf_counter()

    if pending:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_rid: Dict[Any, str] = {}
            for run_id, run_cfg in pending:
                cfg_dict = _config_to_dict(run_cfg)
                future = pool.submit(
                    _run_worker_direct,
                    run_id,
                    cfg_dict,
                    str(output_base),
                    campaign_id,
                )
                future_to_rid[future] = run_id

            n_done = 0
            n_total = len(pending)

            for future in as_completed(future_to_rid):
                run_id = future_to_rid[future]
                n_done += 1

                try:
                    result = future.result()

                    if result.get("status") == "failed":
                        manifest.mark_failed(run_id)
                        failed_runs.append({
                            "run_id": run_id,
                            "error": result.get("error", "unknown"),
                        })
                        logger.warning(
                            "[%d/%d] Run %s FAILED: %s",
                            n_done, n_total, run_id, result.get("error"),
                        )
                    else:
                        manifest.mark_completed(run_id)
                        results.append(result)
                        logger.info(
                            "[%d/%d] Run %s completed (AI=%.4f)",
                            n_done, n_total, run_id,
                            result.get("AI", float("nan")),
                        )

                except Exception as exc:
                    manifest.mark_failed(run_id)
                    failed_runs.append({"run_id": run_id, "error": str(exc)})
                    logger.error(
                        "[%d/%d] Run %s raised exception: %s",
                        n_done, n_total, run_id, exc,
                    )

    wall_time = time.perf_counter() - t0
    logger.info(
        "Campaign %s finished in %.1f s: %d completed, %d failed, %d skipped",
        campaign_id, wall_time, len(results), len(failed_runs), n_skipped,
    )

    # ---- Save failed runs list -------------------------------------------
    if failed_runs:
        failed_path = campaign_dir / "failed_runs.csv"
        save_csv(pd.DataFrame(failed_runs), failed_path)
        logger.info("Failed runs saved to %s", failed_path)

    # ---- Aggregate all results -------------------------------------------
    df = aggregate_results(campaign_id, output_base)
    if failed_runs:
        raise RuntimeError(f"{len(failed_runs)} runs failed; inspect failed_runs.csv")
    return df


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------

_RESULTS_COLUMNS: List[str] = [
    "run_id",
    "frequency",
    "D",
    "geometry",
    "replicate",
    "bridge_width",
    "seed",
    "mean_h",
    "max_h",
    "AI",
    "UI",
    "depletion_depth",
    "timescale_ratio",
    "mean_growth_rate",
    "total_charge_proxy",
    "roughness",
]


def aggregate_results(
    campaign_id: str,
    output_base: Path,
) -> pd.DataFrame:
    """Collect ``metrics.json`` from all completed runs and produce ``results.csv``.

    Parameters
    ----------
    campaign_id : str
    output_base : Path
        Root data directory (contains ``simulated_raw/`` and ``simulated_agg/``).

    Returns
    -------
    pd.DataFrame
        One row per completed run with standardised columns.
    """
    raw_dir = output_base / "simulated_raw" / campaign_id
    agg_dir = ensure_dir(output_base / "simulated_agg" / campaign_id)

    rows: List[Dict[str, Any]] = []

    if raw_dir.exists():
        for run_dir in sorted(raw_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            metrics_path = run_dir / "metrics.json"
            meta_path = run_dir / "metadata.json"

            if not metrics_path.exists():
                continue

            # Only include completed runs
            if meta_path.exists():
                try:
                    meta = load_json(meta_path)
                    if meta.get("status") != "completed":
                        continue
                except (json.JSONDecodeError, KeyError):
                    continue

            try:
                metrics = load_json(metrics_path)
                run_id = metrics.get("run_id", run_dir.name)

                row: Dict[str, Any] = {"run_id": run_id}
                row["geometry"] = metrics.get(
                    "geometry", _infer_geometry_label(metrics.get("bridge_width", 0))
                )
                row["replicate"] = _infer_replicate(run_id)

                for col in _RESULTS_COLUMNS:
                    if col not in row:
                        row[col] = metrics.get(col)

                rows.append(row)

            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(
                    "Could not read metrics for %s: %s", run_dir.name, exc
                )

    df = pd.DataFrame(rows)

    # Ensure standard column ordering; add missing columns as None
    for col in _RESULTS_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[[c for c in _RESULTS_COLUMNS if c in df.columns]]

    results_path = agg_dir / "results.csv"
    save_csv(df, results_path)
    logger.info("Aggregated %d results -> %s", len(df), results_path)

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_geometry_label(bridge_width: float) -> str:
    """Map bridge_width to a human-readable geometry label."""
    if bridge_width <= 0.003:
        return "narrow"
    return "wide"


def _infer_replicate(run_id: str) -> int:
    """Extract replicate index from a factorial run_id like ``f1_Dlow_gnarrow_r2``."""
    parts = run_id.split("_")
    for part in reversed(parts):
        if part.startswith("r") and part[1:].isdigit():
            return int(part[1:])
    return 0
