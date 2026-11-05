"""
Statistical analysis for campaign results.

Performs:
- 3-way factorial ANOVA (frequency x D_level x geometry)
- Multiple linear regression with polynomial terms
- Descriptive statistics per condition
- Campaign-level visualization plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from .constants import D_LEVELS
from .utils_io import ensure_dir, save_csv, save_json

logger = logging.getLogger(__name__)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical column names and derived columns exist.

    Adds ``D_level`` (categorical) from numeric ``D`` column using
    the predefined thresholds in ``constants.D_LEVELS``.
    """
    df = df.copy()

    # Derive D_level from numeric D if missing
    if "D_level" not in df.columns and "D" in df.columns:
        boundaries = sorted(D_LEVELS.values())  # [1e-10, 6.5e-10, 2e-9]
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


def run_anova(df: pd.DataFrame) -> dict:
    """Run 3-way factorial ANOVA for AI and UI.

    Factors: frequency (categorical), D_level (categorical), geometry (categorical).

    Parameters
    ----------
    df : pd.DataFrame
        Results with columns: frequency, D_level, geometry, AI, UI.

    Returns
    -------
    dict
        Keys: 'AI_anova', 'UI_anova' with ANOVA table DataFrames.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        logger.warning("statsmodels not available, skipping ANOVA")
        return {}

    results = {}

    # Ensure categorical
    df = df.copy()
    df["freq_cat"] = df["frequency"].astype(str)
    df["D_cat"] = df["D_level"].astype(str)
    df["geom_cat"] = df["geometry"].astype(str)

    for dv in ["AI", "UI"]:
        if dv not in df.columns:
            continue
        formula = f"{dv} ~ C(freq_cat) * C(D_cat) * C(geom_cat)"
        try:
            model = ols(formula, data=df).fit()
            anova_table = anova_lm(model, typ=2)
            # Compute eta-squared (effect size)
            ss_total = anova_table["sum_sq"].sum()
            anova_table["eta_sq"] = anova_table["sum_sq"] / ss_total
            results[f"{dv}_anova"] = anova_table
            results[f"{dv}_model"] = model
            logger.info("ANOVA for %s: R² = %.4f", dv, model.rsquared)
        except Exception as e:
            logger.error("ANOVA failed for %s: %s", dv, e)

    return results


def run_regression(df: pd.DataFrame) -> dict:
    """Run multiple linear regression for AI and UI.

    Includes polynomial terms for frequency and D.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Regression model summaries and coefficients.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        logger.warning("statsmodels not available, skipping regression")
        return {}

    results = {}
    df = df.copy()

    # Create numeric features
    df["log_freq"] = np.log10(df["frequency"].clip(lower=0.1))
    df["log_D"] = np.log10(df["D"].clip(lower=1e-12))
    df["geom_num"] = (df["geometry"] == "wide").astype(float)
    df["log_freq_sq"] = df["log_freq"] ** 2
    df["log_D_sq"] = df["log_D"] ** 2
    df["freq_D_interaction"] = df["log_freq"] * df["log_D"]

    features = ["log_freq", "log_D", "geom_num", "log_freq_sq", "log_D_sq", "freq_D_interaction"]
    X = sm.add_constant(df[features].values)
    feature_names = ["const"] + features

    for dv in ["AI", "UI"]:
        if dv not in df.columns:
            continue
        y = df[dv].values
        try:
            model = sm.OLS(y, X).fit()
            results[f"{dv}_regression"] = model
            results[f"{dv}_coefficients"] = dict(zip(feature_names, model.params))
            results[f"{dv}_r_squared"] = model.rsquared
            results[f"{dv}_summary"] = model.summary().as_text()
            logger.info("Regression for %s: R² = %.4f", dv, model.rsquared)
        except Exception as e:
            logger.error("Regression failed for %s: %s", dv, e)

    return results


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std of DVs per unique condition."""
    group_cols = [c for c in ["frequency", "D_level", "geometry"] if c in df.columns]
    dv_cols = [c for c in ["AI", "UI", "depletion_depth", "timescale_ratio", "mean_growth_rate"]
               if c in df.columns]
    if not group_cols or not dv_cols:
        return pd.DataFrame()

    stats = df.groupby(group_cols)[dv_cols].agg(["mean", "std"]).reset_index()
    stats.columns = ["_".join(c).strip("_") for c in stats.columns]
    return stats


def save_anova_results(anova_results: dict, output_dir: Path) -> None:
    """Save ANOVA tables to files."""
    output_dir = ensure_dir(Path(output_dir))
    for key, table in anova_results.items():
        if isinstance(table, pd.DataFrame):
            save_csv(table.reset_index(), output_dir / f"{key}_table.csv")
        elif hasattr(table, "summary"):
            with open(output_dir / f"{key}_summary.txt", "w") as f:
                f.write(str(table.summary()))


def save_regression_results(regression_results: dict, output_dir: Path) -> None:
    """Save regression results."""
    output_dir = ensure_dir(Path(output_dir))
    for key, val in regression_results.items():
        if key.endswith("_summary"):
            with open(output_dir / f"{key}.txt", "w") as f:
                f.write(str(val))
        elif key.endswith("_coefficients"):
            save_json(val, output_dir / f"{key}.json")


# --- Campaign-level plots ---

def plot_ai_vs_frequency(df: pd.DataFrame, save_dir: Path, dpi: int = 300) -> None:
    """AI vs frequency grouped by D and geometry."""
    save_dir = ensure_dir(Path(save_dir))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, geom in enumerate(df["geometry"].unique()):
        ax = axes[idx] if len(df["geometry"].unique()) > 1 else axes[0]
        sub = df[df["geometry"] == geom]
        for d_level in sub["D_level"].unique():
            dd = sub[sub["D_level"] == d_level]
            means = dd.groupby("frequency")["AI"].mean()
            stds = dd.groupby("frequency")["AI"].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker="o", label=f"D={d_level}", capsize=4)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Anisotropy Index (AI)")
        ax.set_title(f"Geometry: {geom}")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "ai_vs_frequency.png", dpi=dpi)
    plt.close(fig)


def plot_ui_vs_frequency(df: pd.DataFrame, save_dir: Path, dpi: int = 300) -> None:
    """UI vs frequency grouped by D and geometry."""
    save_dir = ensure_dir(Path(save_dir))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for idx, geom in enumerate(df["geometry"].unique()):
        ax = axes[idx] if len(df["geometry"].unique()) > 1 else axes[0]
        sub = df[df["geometry"] == geom]
        for d_level in sub["D_level"].unique():
            dd = sub[sub["D_level"] == d_level]
            means = dd.groupby("frequency")["UI"].mean()
            stds = dd.groupby("frequency")["UI"].std()
            ax.errorbar(means.index, means.values, yerr=stds.values,
                        marker="s", label=f"D={d_level}", capsize=4)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Uniformity Index (UI) [m]")
        ax.set_title(f"Geometry: {geom}")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "ui_vs_frequency.png", dpi=dpi)
    plt.close(fig)


def plot_heatmap_ai(df: pd.DataFrame, save_dir: Path, dpi: int = 300) -> None:
    """Heatmap of AI(frequency, D_level) per geometry."""
    save_dir = ensure_dir(Path(save_dir))

    for geom in df["geometry"].unique():
        sub = df[df["geometry"] == geom]
        pivot = sub.pivot_table(index="D_level", columns="frequency", values="AI", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{f:.0f}" for f in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("D level")
        ax.set_title(f"Anisotropy Index - {geom} geometry")
        plt.colorbar(im, ax=ax, label="AI")
        # Annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
        fig.savefig(save_dir / f"heatmap_ai_{geom}.png", dpi=dpi)
        plt.close(fig)


def plot_heatmap_ui(df: pd.DataFrame, save_dir: Path, dpi: int = 300) -> None:
    """Heatmap of UI(frequency, D_level) per geometry."""
    save_dir = ensure_dir(Path(save_dir))

    for geom in df["geometry"].unique():
        sub = df[df["geometry"] == geom]
        pivot = sub.pivot_table(index="D_level", columns="frequency", values="UI", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{f:.0f}" for f in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("D level")
        ax.set_title(f"Uniformity Index - {geom} geometry")
        plt.colorbar(im, ax=ax, label="UI [m]")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i, j]:.2e}", ha="center", va="center", fontsize=8)
        fig.savefig(save_dir / f"heatmap_ui_{geom}.png", dpi=dpi)
        plt.close(fig)


def plot_response_surface(df: pd.DataFrame, dv: str, save_dir: Path, dpi: int = 300) -> None:
    """3D response surface of DV(frequency, D) per geometry."""
    save_dir = ensure_dir(Path(save_dir))

    for geom in df["geometry"].unique():
        sub = df[df["geometry"] == geom]
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(np.log10(sub["frequency"]), np.log10(sub["D"]), sub[dv],
                   c=sub[dv], cmap="viridis", s=50, alpha=0.8)
        ax.set_xlabel("log₁₀(Frequency [Hz])")
        ax.set_ylabel("log₁₀(D [m²/s])")
        ax.set_zlabel(dv)
        ax.set_title(f"{dv} Response Surface - {geom}")
        fig.savefig(save_dir / f"response_surface_{dv.lower()}_{geom}.png", dpi=dpi)
        plt.close(fig)


def plot_sensitivity_tornado(regression_results: dict, save_dir: Path, dpi: int = 300) -> None:
    """Tornado plot of regression coefficients for sensitivity."""
    save_dir = ensure_dir(Path(save_dir))

    for dv in ["AI", "UI"]:
        key = f"{dv}_coefficients"
        if key not in regression_results:
            continue
        coeffs = regression_results[key]
        # Remove constant
        coeffs = {k: v for k, v in coeffs.items() if k != "const"}
        if not coeffs:
            continue

        names = list(coeffs.keys())
        values = list(coeffs.values())
        abs_vals = [abs(v) for v in values]

        # Sort by absolute value
        sorted_idx = np.argsort(abs_vals)
        names = [names[i] for i in sorted_idx]
        values = [values[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["steelblue" if v > 0 else "coral" for v in values]
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Coefficient")
        ax.set_title(f"Sensitivity Analysis - {dv}")
        plt.tight_layout()
        fig.savefig(save_dir / f"sensitivity_tornado_{dv.lower()}.png", dpi=dpi)
        plt.close(fig)


def plot_scatter_3d_params(df: pd.DataFrame, save_dir: Path, dpi: int = 300) -> None:
    """3D scatter of all runs in parameter space colored by AI."""
    save_dir = ensure_dir(Path(save_dir))
    if "AI" not in df.columns:
        return

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        np.log10(df["frequency"].clip(lower=0.1)),
        np.log10(df["D"].clip(lower=1e-12)),
        df.get("mean_growth_rate", np.zeros(len(df))),
        c=df["AI"], cmap="RdYlGn_r", s=50, alpha=0.8,
    )
    ax.set_xlabel("log₁₀(f [Hz])")
    ax.set_ylabel("log₁₀(D [m²/s])")
    ax.set_zlabel("Mean growth rate")
    plt.colorbar(sc, ax=ax, label="AI", shrink=0.6)
    ax.set_title("Parameter Space Exploration")
    fig.savefig(save_dir / "scatter_3d_params.png", dpi=dpi)
    plt.close(fig)


def plot_main_effects(df: pd.DataFrame, save_dir: Path, dpi: int = 300) -> None:
    """Main effects and interaction plots for ANOVA."""
    save_dir = ensure_dir(Path(save_dir))

    for dv in ["AI", "UI"]:
        if dv not in df.columns:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Main effect: frequency
        means = df.groupby("frequency")[dv].mean()
        axes[0].plot(means.index, means.values, "o-", linewidth=2)
        axes[0].set_xlabel("Frequency [Hz]")
        axes[0].set_ylabel(f"Mean {dv}")
        axes[0].set_title(f"Main Effect: Frequency on {dv}")
        axes[0].set_xscale("log")

        # Main effect: D_level
        means = df.groupby("D_level")[dv].mean()
        axes[1].bar(means.index, means.values, color="steelblue")
        axes[1].set_xlabel("D level")
        axes[1].set_ylabel(f"Mean {dv}")
        axes[1].set_title(f"Main Effect: D on {dv}")

        # Main effect: geometry
        means = df.groupby("geometry")[dv].mean()
        axes[2].bar(means.index, means.values, color="coral")
        axes[2].set_xlabel("Geometry")
        axes[2].set_ylabel(f"Mean {dv}")
        axes[2].set_title(f"Main Effect: Geometry on {dv}")

        plt.tight_layout()
        fig.savefig(save_dir / f"main_effects_{dv.lower()}.png", dpi=dpi)
        plt.close(fig)


def run_full_analysis(campaign_id: str, output_base: Path) -> None:
    """Orchestrate full statistical analysis for a campaign."""
    output_base = Path(output_base)
    campaign_dir = output_base / "simulated_agg" / campaign_id
    results_path = campaign_dir / "results.csv"

    if not results_path.exists():
        logger.error("Results file not found: %s", results_path)
        return

    df = _normalize_columns(pd.read_csv(results_path))
    logger.info("Loaded %d rows from %s", len(df), results_path)

    fig_dir = ensure_dir(output_base / "reports" / campaign_id / "figures")

    # Descriptive statistics
    desc = compute_descriptive_stats(df)
    if not desc.empty:
        save_csv(desc, campaign_dir / "descriptive_stats.csv")

    # ANOVA
    anova_results = run_anova(df)
    save_anova_results(anova_results, campaign_dir)

    # Regression
    regression_results = run_regression(df)
    save_regression_results(regression_results, campaign_dir)

    # Plots
    try:
        plot_ai_vs_frequency(df, fig_dir)
        plot_ui_vs_frequency(df, fig_dir)
        plot_heatmap_ai(df, fig_dir)
        plot_heatmap_ui(df, fig_dir)
        plot_response_surface(df, "AI", fig_dir)
        plot_response_surface(df, "UI", fig_dir)
        plot_main_effects(df, fig_dir)
        plot_sensitivity_tornado(regression_results, fig_dir)
        plot_scatter_3d_params(df, fig_dir)
    except Exception as e:
        logger.error("Plot generation failed: %s", e)

    logger.info("Full analysis complete for campaign %s", campaign_id)
