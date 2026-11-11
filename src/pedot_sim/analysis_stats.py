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
    """Label only exact design levels; never bin continuous diffusion values."""
    df = df.copy()
    if "D_level" not in df and "D" in df:
        df["D_level"] = df["D"].map({v: k for k, v in D_LEVELS.items()}).fillna("continuous")
    return df


CONDITION_COLUMNS = ["frequency", "D", "bridge_width", "geometry", "V_DC", "V_AC",
                     "waveform_type", "duty_cycle", "adc_ratio", "rest_fraction", "physics_stage", "t_end"]


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

    if any(df[c].nunique() < 2 for c in ("frequency", "D_level", "geometry")):
        raise ValueError("Factorial ANOVA requires multiple levels for all three factors")
    if len(df) <= df.groupby(["frequency", "D_level", "geometry"]).ngroups:
        raise ValueError("ANOVA requires replicated conditions, not a continuous dense sweep")
    if "noise_enabled" not in df or not df["noise_enabled"].eq(True).all():
        raise ValueError("ANOVA requires independent stochastic replicates; deterministic reruns are not replicates")
    if (df["D_level"] == "continuous").any():
        raise ValueError("Continuous sweeps are not categorical factorial experiments")
    factors = ["frequency", "D_level", "geometry"]
    groups = df.groupby(factors)
    expected = np.prod([df[c].nunique() for c in factors])
    if groups.ngroups != expected or groups.size().min() < 2:
        raise ValueError("ANOVA requires a complete factorial design with >=2 replicates per condition")
    for col in CONDITION_COLUMNS:
        if col not in factors and col not in ("D", "bridge_width") and col in df and df[col].nunique() > 1:
            raise ValueError(f"Unmodeled varying factor in factorial ANOVA: {col}")
    if "seed" not in df or df["seed"].nunique() != len(df):
        raise ValueError("Independent random seeds are required for every factorial run")
    results = {}

    # Ensure categorical
    df = df.copy()
    df["freq_cat"] = df["frequency"].astype(str)
    df["D_cat"] = df["D_level"].astype(str)
    df["geom_cat"] = df["geometry"].astype(str)

    for dv in ["AI", "UI"]:
        if dv not in df.columns:
            continue
        residual_ss = float(np.sum((df[dv]-groups[dv].transform("mean"))**2))
        scale = max(float(np.sum(df[dv]**2)), np.finfo(float).tiny)
        if residual_ss <= 1e-24*scale:
            raise ValueError(f"{dv} has no estimable within-condition variance")
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
            raise ValueError(f"ANOVA failed for {dv}") from e

    return results


def run_regression(df: pd.DataFrame) -> dict:
    """Descriptive standardized linear surrogate; no causal or validation claim.

    Uses actual numeric bridge width and voltage parameters, excludes constants,
    rejects confounding/rank deficiency and reports leave-one-condition-out RMSE.
    Inferential p-values from simulation surrogates are deliberately not emitted.
    """
    features = pd.DataFrame(index=df.index)
    for name in ("frequency", "D", "bridge_width", "V_DC", "V_AC"):
        if name not in df or df[name].nunique() < 2:
            continue
        values = df[name].astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Regression inputs must be finite")
        if name in ("frequency", "D"):
            if (values <= 0).any():
                raise ValueError("Log predictors require positive frequency and D")
            features["log10_"+name] = np.log10(values)
        else:
            features[name] = values
    if features.empty:
        raise ValueError("No varying regression predictors")
    means, scales = features.mean(), features.std(ddof=0)
    X = np.column_stack([np.ones(len(df)), ((features-means)/scales).to_numpy()])
    if len(df) <= X.shape[1] or np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Regression requires a full rank design and residual degrees of freedom")
    names = ["const"]+list(features.columns)
    conditions = features.astype(str).agg("|".join, axis=1).to_numpy()
    results = {"predictor_scaling": {name: {"mean": float(means[name]), "std": float(scales[name])} for name in features}}
    for dv in ("AI", "UI", "mean_h", "mean_growth_rate"):
        if dv not in df:
            continue
        y = df[dv].to_numpy(float)
        if not np.isfinite(y).all():
            raise ValueError("Regression outcomes must be finite")
        coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = X@coefficients-y
        ss = float(np.sum((y-y.mean())**2))
        r2 = 1-float(residuals@residuals)/ss if ss>0 else None
        held_out = np.full(len(y), np.nan)
        for condition in np.unique(conditions):
            test = conditions == condition
            train = ~test
            if train.sum() > X.shape[1] and np.linalg.matrix_rank(X[train]) == X.shape[1]:
                beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
                held_out[test] = X[test]@beta
        cv_rmse = float(np.sqrt(np.mean((held_out-y)**2))) if np.isfinite(held_out).all() else None
        results[dv+"_coefficients"] = dict(zip(names, map(float, coefficients)))
        results[dv+"_r_squared"] = r2
        results[dv+"_cv_rmse"] = cv_rmse
        results[dv+"_summary"] = (f"Descriptive linear surrogate; standardized predictors. R2={r2}; "
            f"leave-one-condition-out RMSE={cv_rmse}. Coefficients are conditional associations, "
            "not causal sensitivities. Cross-validation uses simulation data, not experiments.")
    return results


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std of DVs per unique condition."""
    group_cols = [c for c in CONDITION_COLUMNS if c in df.columns]
    dv_cols = [c for c in ["AI", "UI", "mean_h", "depletion_depth", "mean_growth_rate", "thickness_cv", "deposited_mass_kg_per_m", "electrolyte_ohmic_energy_J_per_m"]
               if c in df.columns]
    if not group_cols or not dv_cols:
        return pd.DataFrame()

    stats = df.groupby(group_cols)[dv_cols].agg(["count", "mean", "std", "min", "max"]).reset_index()
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
        ax.set_xlabel("Outcome change per predictor standard deviation")
        ax.set_title(f"Descriptive standardized associations - {dv}")
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


def run_full_analysis(campaign_id: str, output_base: Path) -> dict:
    """Analyze any campaign; record why an inferential method is inapplicable."""
    from .utils_io import validate_id
    validate_id(campaign_id)
    output_base = Path(output_base)
    campaign_dir = output_base / "simulated_agg" / campaign_id
    df = _normalize_columns(pd.read_csv(campaign_dir / "results.csv"))
    if df.empty:
        raise ValueError("No completed runs to analyze")
    for col in ("AI", "UI", "mean_h", "frequency", "D"):
        if col not in df or not np.isfinite(df[col]).all():
            raise ValueError(f"Missing/nonfinite analysis data: {col}")
    fig_dir = ensure_dir(output_base / "reports" / campaign_id / "figures")
    save_csv(compute_descriptive_stats(df), campaign_dir / "descriptive_stats.csv")
    decisions = {"interpretation": "Exploratory simulation comparisons; no empirical validation or manufacturing optimum established."}
    for name, compute, save in (("anova", run_anova, save_anova_results),
                                ("regression", run_regression, save_regression_results)):
        # Delete stale results when re-analysis makes a method inapplicable.
        for stale in campaign_dir.glob("*"):
            if name in stale.name and stale.suffix in (".csv", ".txt", ".json"):
                stale.unlink()
        try:
            result = compute(df)
            save(result, campaign_dir)
            decisions[name] = {"status": "computed"}
            if name == "regression":
                save_json(result, campaign_dir / "regression_diagnostics.json")
                plot_sensitivity_tornado(result, fig_dir)
        except ValueError as exc:
            decisions[name] = {"status": "not_applicable", "reason": str(exc)}
    # Scatter displays individual continuous designs without invented categories.
    plot_scatter_3d_params(df, fig_dir, dpi=150)
    save_json(decisions, campaign_dir / "analysis_decisions.json")
    return decisions
