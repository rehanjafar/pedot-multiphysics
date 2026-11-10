"""
Optimization and Pareto front analysis.

Defines multi-objective optimization:
- Minimize UI (uniformity)
- Control AI (target range)
- Maximize surface area (roughness proxy)

Computes Pareto front and identifies best conditions.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import D_LEVELS
from .utils_io import ensure_dir, save_csv, save_json

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure D_level column exists (derived from numeric D)."""
    df = df.copy()
    if "D_level" not in df.columns and "D" in df.columns:
        labels_map = {v: k for k, v in D_LEVELS.items()}

        def _assign_level(d_val: float) -> str:
            best_label = "mid"
            best_dist = float("inf")
            for d_ref, label in labels_map.items():
                dist = abs(d_val - d_ref)
                if dist < best_dist:
                    best_dist = dist
                    best_label = label
            return best_label

        df["D_level"] = df["D"].apply(_assign_level)
    return df


def compute_objective_scores(
    df: pd.DataFrame,
    w_uniformity: float = 0.5,
    w_anisotropy: float = 0.3,
    w_roughness: float = 0.2,
    ai_target: float = 1.0,
) -> pd.DataFrame:
    """Add normalized objective scores to results.

    Score = w1 * (1/UI_norm) + w2 * AI_target_proximity + w3 * roughness_norm

    Higher score = better.

    Parameters
    ----------
    df : pd.DataFrame
    w_uniformity, w_anisotropy, w_roughness : float
        Objective weights.
    ai_target : float
        Target AI (1.0 = perfectly uniform).

    Returns
    -------
    pd.DataFrame
        With added 'score' column.
    """
    df = df.copy()
    required = [c for c in ("AI", "UI", "mean_growth_rate") if c in df]
    if required and not np.isfinite(df[required].values).all():
        raise ValueError("Optimization requires finite metrics")
    if "mean_h" in df:
        df = df[df.mean_h > 0].copy()
    if df.empty:
        raise ValueError("Optimization requires a deposited film")

    # Normalize UI (lower is better -> invert)
    if "UI" in df.columns and df["UI"].max() > 0:
        ui_norm = df["UI"] / df["UI"].max()
        df["UI_score"] = 1.0 - ui_norm
    else:
        df["UI_score"] = 1.0

    # AI proximity to target (closer to target is better)
    if "AI" in df.columns:
        ai_dist = np.abs(df["AI"] - ai_target)
        ai_max_dist = ai_dist.max() if ai_dist.max() > 0 else 1.0
        df["AI_score"] = 1.0 - ai_dist / ai_max_dist
    else:
        df["AI_score"] = 1.0

    # Roughness (higher can be better for surface area)
    if "roughness" in df.columns and df["roughness"].max() > 0:
        df["roughness_score"] = df["roughness"] / df["roughness"].max()
    else:
        df["roughness_score"] = 0.0

    df["score"] = (
        w_uniformity * df["UI_score"]
        + w_anisotropy * df["AI_score"]
        + w_roughness * df["roughness_score"]
    )

    return df


def compute_pareto_front(
    df: pd.DataFrame,
    obj1: str = "UI",
    obj2: str = "mean_growth_rate",
    minimize_obj1: bool = True,
    maximize_obj2: bool = True,
) -> pd.DataFrame:
    """Identify Pareto-optimal solutions.

    Parameters
    ----------
    df : pd.DataFrame
    obj1, obj2 : str
        Column names for the two objectives.
    minimize_obj1 : bool
        True if obj1 should be minimized.
    maximize_obj2 : bool
        True if obj2 should be maximized.

    Returns
    -------
    pd.DataFrame
        Subset of df on the Pareto front.
    """
    if obj1 not in df.columns or obj2 not in df.columns:
        logger.warning("Objective columns missing for Pareto analysis")
        return pd.DataFrame()

    df = df.copy()
    # Convert to minimization
    v1 = df[obj1].values.copy()
    v2 = df[obj2].values.copy()
    if not minimize_obj1:
        v1 = -v1
    if maximize_obj2:
        v2 = -v2

    n = len(df)
    is_pareto = np.ones(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= in all objectives and < in at least one
            if v1[j] <= v1[i] and v2[j] <= v2[i]:
                if v1[j] < v1[i] or v2[j] < v2[i]:
                    is_pareto[i] = False
                    break

    return df[is_pareto].copy()


def plot_pareto(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    save_path: Path,
    obj1: str = "UI",
    obj2: str = "mean_growth_rate",
    dpi: int = 300,
) -> None:
    """Plot Pareto front with all points."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by AI
    if "AI" in df.columns:
        sc = ax.scatter(df[obj1], df[obj2], c=df["AI"], cmap="RdYlGn_r",
                        s=40, alpha=0.5, label="All runs")
        plt.colorbar(sc, ax=ax, label="AI")
    else:
        ax.scatter(df[obj1], df[obj2], s=40, alpha=0.5, label="All runs")

    # Pareto front
    if not pareto_df.empty:
        pareto_sorted = pareto_df.sort_values(obj1)
        ax.plot(pareto_sorted[obj1], pareto_sorted[obj2], "r-", linewidth=2, alpha=0.8)
        ax.scatter(pareto_sorted[obj1], pareto_sorted[obj2],
                   c="red", s=100, marker="*", zorder=5, label="Pareto front")

    ax.set_xlabel(f"{obj1}")
    ax.set_ylabel(f"{obj2}")
    ax.set_title("Pareto Front: Uniformity vs Growth Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(save_path), dpi=dpi)
    plt.close(fig)


def find_best_conditions(df: pd.DataFrame) -> pd.DataFrame:
    """Find best conditions by composite score."""
    if "score" not in df.columns:
        df = compute_objective_scores(df)
    cols = ["run_id", "frequency", "geometry", "AI", "UI", "score"]
    if "D_level" in df.columns:
        cols.insert(2, "D_level")
    available = [c for c in cols if c in df.columns]
    return df.nlargest(5, "score")[available].copy()


def save_optimization_results(
    df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    best_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save optimization outputs."""
    output_dir = ensure_dir(Path(output_dir))
    if not pareto_df.empty:
        save_csv(pareto_df, output_dir / "pareto_front.csv")
    save_csv(best_df, output_dir / "best_conditions.csv")


def run_optimization_analysis(campaign_id: str, output_base: Path) -> None:
    """Run full optimization analysis."""
    output_base = Path(output_base)
    campaign_dir = output_base / "simulated_agg" / campaign_id
    results_path = campaign_dir / "results.csv"

    if not results_path.exists():
        logger.error("Results not found: %s", results_path)
        return

    df = _normalize_columns(pd.read_csv(results_path))
    df = compute_objective_scores(df)

    fig_dir = ensure_dir(output_base / "reports" / campaign_id / "figures")

    pareto_df = compute_pareto_front(df)
    best_df = find_best_conditions(df)

    plot_pareto(df, pareto_df, fig_dir / "pareto_front.png")
    save_optimization_results(df, pareto_df, best_df, campaign_dir)

    logger.info("Optimization analysis complete: %d Pareto-optimal runs", len(pareto_df))
