"""Purposeful graphs from archived numerical values, with data lineage and units."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .figure_style import (
    BLUE,
    ORANGE,
    TEAL,
    scientific_style,
    save_figure,
    save_interactive,
    write_gallery,
)
from .viz3d import network_3d


def thickness_history_surface(x, times, history, directory):
    x, times, history = np.asarray(x), np.asarray(times), np.asarray(history)
    if history.shape != (len(times), len(x)) or len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Thickness history needs matching data and increasing real timestamps")
    X, T = np.meshgrid(x * 1e3, times)
    with scientific_style():
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        im = ax.plot_surface(
            X, T, history * 1e6, cmap="viridis", rcount=len(times), ccount=len(x), linewidth=0
        )
        ax.set(
            xlabel="Electrode position [mm]",
            ylabel="Time [s]",
            zlabel="Thickness [µm]",
            title="Where and when does the film grow?",
        )
        ax.view_init(28, -65)
        fig.colorbar(im, ax=ax, shrink=0.55, pad=0.12, label="Model thickness [µm]")
        entry = save_figure(
            fig,
            directory,
            "thickness_history_3d",
            "Where and when does the film grow?",
            "Surface connects saved spatial profiles at their actual timestamps. The second horizontal axis is time, not a physical depth. No unrecorded transient is recovered.",
            "fixed-strip model",
        )
    entry["interactive"] = save_interactive(
        go.Figure(
            go.Surface(
                x=x * 1e3,
                y=times,
                z=history * 1e6,
                colorscale="Viridis",
                colorbar={"title": "h [µm]"},
            )
        ),
        directory,
        "thickness_history_3d",
        "Thickness in space and time",
        "A field history, not a three-dimensional reconstruction.",
        {
            "xaxis_title": "Position [mm]",
            "yaxis_title": "Time [s]",
            "zaxis_title": "Thickness [µm]",
        },
    )
    pd.DataFrame(
        {"x_m": X.ravel() * 1e-3, "time_s": T.ravel(), "thickness_m": history.ravel()}
    ).to_csv(Path(directory) / "thickness_history.csv", index=False)
    return entry


def dendrite_figures(result, config, run_dir):
    root = Path(run_dir) / "views"
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    owner = result["owner"]
    initial = result["initial_owner"]
    palette = ListedColormap(["#F6F8FA", ORANGE, BLUE])
    records = pd.DataFrame(result["records"])
    with scientific_style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for ax, field, title in zip(
            axes, [initial, owner], ["Initial wires", "Final conducting network"]
        ):
            ax.imshow(field, origin="lower", cmap=palette, vmin=0, vmax=2, interpolation="nearest")
            ax.set(xlabel="x [lattice cells]", ylabel="y [lattice cells]", title=title)
        entries.append(
            save_figure(
                fig,
                root,
                "morphology_comparison",
                "Where did the aggregate material attach?",
                f"Orange: driven network; blue: grounded network. Seed {config.seed}; elapsed {result['metrics']['elapsed_model_steps']:.3g} model steps. Gold and connected deposits share their electrical label.",
                "uncalibrated direct-wire model",
            )
        )
        count = min(4, len(result["snapshots"]))
        indices = np.unique(np.linspace(0, len(result["snapshots"]) - 1, count, dtype=int))
        fig, axes = plt.subplots(
            1, len(indices), figsize=(3.4 * len(indices), 3.7), layout="constrained", squeeze=False
        )
        for ax, index in zip(axes.ravel(), indices):
            ax.imshow(
                result["snapshots"][index],
                origin="lower",
                cmap=palette,
                vmin=0,
                vmax=2,
                interpolation="nearest",
            )
            ax.set(
                xlabel="x [cells]",
                ylabel="y [cells]",
                title=f"t = {result['snapshot_times'][index]:.3g} model steps",
            )
        entries.append(
            save_figure(
                fig,
                root,
                "growth_sequence",
                "How did the networks develop?",
                "Selected recorded snapshots, with their actual timestamps; no intermediate branch shapes are synthesized.",
                "recorded model states",
            )
        )
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
        axes[0].plot(records.step, records.left_deposited_cells, color=ORANGE, label="Driven")
        axes[0].plot(records.step, records.right_deposited_cells, color=BLUE, label="Grounded")
        axes[0].set(ylabel="Deposited cells", title="Material attachment")
        axes[0].legend(frameon=False)
        axes[1].plot(records.step, records.mobile_aggregates, color=TEAL)
        axes[1].set(ylabel="Mobile aggregates", title="Remaining mobile inventory")
        axes[2].plot(records.step, records.area_asymmetry, color=BLUE)
        axes[2].axhline(0, color="#8A9BA9", lw=1)
        axes[2].set(
            ylim=(-1, 1), ylabel="(driven − grounded) / total", title="Projected area imbalance"
        )
        for ax in axes:
            ax.set_xlabel("Time [model steps]")
            ax.grid(axis="y")
        entries.append(
            save_figure(
                fig,
                root,
                "growth_diagnostics",
                "How much grows on each wire?",
                "All integration records are shown. Deposited + mobile aggregates is conserved. Undefined imbalance before growth is left missing, not assigned a symmetric value.",
                "model diagnostics",
            )
        )
        if "potential_V" in result:
            potential = result["potential_V"]
            field_owner = result["potential_owner"]
            ey, ex = np.gradient(-potential)
            field = np.ma.masked_where(field_owner > 0, np.hypot(ex, ey))
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
            limit = max(abs(float(potential.min())), abs(float(potential.max())), 1e-12)
            im = axes[0].imshow(
                potential,
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            fig.colorbar(im, ax=axes[0], shrink=0.8, label="Solution/network potential [V]")
            im = axes[1].imshow(field, origin="lower", cmap="magma", interpolation="nearest")
            fig.colorbar(im, ax=axes[1], shrink=0.8, label="Discrete field magnitude [V/cell]")
            for ax, title in zip(
                axes, ["Electrical potential", "Field concentration in the liquid"]
            ):
                ax.set(xlabel="x [lattice cells]", ylabel="y [lattice cells]", title=title)
            entries.append(
                save_figure(
                    fig,
                    root,
                    "electrical_fields",
                    "Where does the electric field concentrate?",
                    f"Last solved pre-attachment geometry at interval midpoint t={result['potential_time']:.4g} model steps. Spatial derivatives are in volts per lattice cell, not V/m. A shorted final geometry is never solved.",
                    "model field, before last attachment",
                )
            )
            fig = plt.figure(figsize=(9, 6), layout="constrained")
            ax = fig.add_subplot(projection="3d")
            y, x = np.indices(potential.shape)
            surface = ax.plot_surface(
                x,
                y,
                potential,
                cmap="RdBu_r",
                rcount=len(y),
                ccount=potential.shape[1],
                linewidth=0,
                vmin=-limit,
                vmax=limit,
            )
            ax.set(
                xlabel="x [cells]",
                ylabel="y [cells]",
                zlabel="Potential [V]",
                title="Potential landscape · height encodes volts",
            )
            fig.colorbar(surface, ax=ax, shrink=0.55, pad=0.12, label="Potential [V]")
            entry = save_figure(
                fig,
                root,
                "potential_3d",
                "How does potential vary across the cell?",
                "Surface height represents electrical potential. It is not electrode thickness or physical elevation. Geometry is the last solved pre-attachment state.",
                "scalar field in 3D",
            )
            entry["interactive"] = save_interactive(
                go.Figure(
                    go.Surface(
                        z=potential,
                        colorscale="RdBu",
                        reversescale=True,
                        cmin=-limit,
                        cmax=limit,
                        colorbar={"title": "Potential [V]"},
                    )
                ),
                root,
                "potential_3d",
                "Electrical potential landscape",
                "Height is volts; x/y are lattice coordinates. Last solved pre-attachment geometry.",
                {
                    "xaxis_title": "x [cells]",
                    "yaxis_title": "y [cells]",
                    "zaxis_title": "Potential [V]",
                },
            )
            entries.append(entry)
    entries.append(network_3d(owner, root))
    return write_gallery(
        root,
        "Direct-wire growth: morphology and fields",
        "Inspectable model outputs with conserved aggregate counts, recorded timing and explicit geometric assumptions.",
        entries,
        [
            "The aggregate model is uncalibrated; lattice cells and model steps are not metres or seconds.",
            "The 3D network is an illustrative extrusion. Potential surfaces encode volts, not spatial height.",
            "One seed is one realization. Use independent ensembles and timestep sensitivity studies before interpreting a trend.",
            "No electrode current, polymer mass, conductivity or manufacturing yield is inferred from these maps.",
        ],
        [
            ("Parameters", "../params.json"),
            ("Full timeseries", "../timeseries.csv"),
            ("Morphology and solved field arrays", "../morphology.npz"),
            ("Illustrative network OBJ", "network_illustrative.obj"),
            ("3D assumptions", "network_3d_metadata.json"),
            ("Run metadata", "../metadata.json"),
        ],
    )


def strip_figures(run_dir):
    root = Path(run_dir)
    out = root / "views"
    out.mkdir(exist_ok=True)
    with np.load(root / "final_state.npz") as archive:
        state = dict(archive)
    params = json.loads((root / "params.json").read_text())
    ts = pd.read_csv(root / "timeseries.csv")
    x, y = state["grid_x"], state["grid_y"]
    wx, wy = state["widths_x"], state["widths_y"]
    xe = np.r_[x[0] - wx[0] / 2, x + wx / 2] * 1e3
    ye = np.r_[y[0] - wy[0] / 2, y + wy / 2] * 1e3
    C = np.ma.masked_where(~state["domain_mask"], state["C"])
    entries = []
    with scientific_style():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), layout="constrained")
        for ax in axes:
            im = ax.pcolormesh(
                xe,
                ye,
                C,
                shading="flat",
                cmap="viridis",
                vmin=0,
                vmax=params["physics"]["C_bulk"],
                rasterized=True,
            )
            ax.set(xlabel="x [mm]", ylabel="y [mm]")
        axes[0].set_title("Entire electrolyte domain")
        ex = state["x"]
        ey = y[np.where(state["electrode_mask"])[0][0]]
        halfwidth = params["geometry"]["bridge_width"] / 2
        axes[1].set(
            xlim=((ex.min() - 0.0003) * 1e3, (ex.max() + 0.0003) * 1e3),
            ylim=((ey - min(0.0003, halfwidth)) * 1e3, (ey + min(0.0003, halfwidth)) * 1e3),
            title="Reaction strip: boundary-layer detail",
        )
        fig.colorbar(im, ax=list(axes), shrink=0.8, label="Monomer concentration [mol/m³]")
        entries.append(
            save_figure(
                fig,
                out,
                "concentration_detail",
                "Is the depleted boundary layer resolved?",
                "Actual finite-volume edges are used, including the strongly graded cells. Full-domain and local views share one color scale. The raw archive retains every concentration value.",
                "fixed-strip model",
            )
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        axes[0].plot(state["x"] * 1e3, state["h"] * 1e6, color=BLUE)
        weights = state["widths_x"][np.where(state["electrode_mask"])[1]]
        mean = np.average(state["h"], weights=weights)
        axes[0].axhline(mean * 1e6, color=ORANGE, ls="--", label="Surface-weighted mean")
        axes[0].set(
            xlabel="Electrode position [mm]",
            ylabel="Thickness [µm]",
            title="Final thickness profile",
        )
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].plot(ts.time, ts.mean_h * 1e6, label="Weighted mean")
        axes[1].plot(ts.time, ts.max_h * 1e6, label="Maximum", color=ORANGE)
        axes[1].set(xlabel="Time [s]", ylabel="Thickness [µm]", title="Growth history")
        axes[1].legend(frameon=False)
        for ax in axes:
            ax.grid(axis="y")
        entries.append(
            save_figure(
                fig,
                out,
                "thickness_detail",
                "How uniform is the deposited film?",
                "Thickness is a mass-consistent model proxy. Surface averages use cell widths. Maximum and mean are distinct; neither identifies a true three-dimensional dendrite.",
                "fixed-strip model",
            )
        )
        if (root / "integration_diagnostics.csv").exists():
            budget = pd.read_csv(root / "integration_diagnostics.csv")
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), layout="constrained")
            axes[0].plot(
                budget.time_end_s, budget.consumed_mol_per_m * 1e6, color=BLUE, label="Consumed"
            )
            axes[0].plot(
                budget.time_end_s,
                budget.reservoir_exchange_mol_per_m * 1e6,
                color=TEAL,
                label="Net reservoir input",
            )
            axes[0].set(ylabel="Cumulative amount [µmol/m depth]", title="Material accounting")
            axes[0].legend(frameon=False, fontsize=8)
            axes[1].plot(budget.time_end_s, budget.balance_error_mol_per_m, color=ORANGE)
            axes[1].set(
                ylabel="Inventory residual [mol/m depth]", title="Independent balance residual"
            )
            axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axes[2].plot(budget.time_end_s, budget.applied_voltage_midpoint_V, color=BLUE)
            axes[2].set(ylabel="Applied drive [V]", title="Resolved interval drive")
            for ax in axes:
                ax.set_xlabel("Time [s]")
                ax.grid(axis="y")
            entries.append(
                save_figure(
                    fig,
                    out,
                    "material_balance",
                    "Does every reaction step conserve material?",
                    "Every integration interval is retained in the accompanying CSV. The residual compares independently integrated reservoir input with total inventory and consumed material. Stage A does not depend on voltage.",
                    "numerical audit",
                )
            )
    if (root / "thickness.npz").exists():
        with np.load(root / "thickness.npz") as hfile:
            times = hfile["snapshot_times"]
            keys = sorted(k for k in hfile.files if k.startswith("h_"))
            if len(times) >= 2:
                entries.append(
                    thickness_history_surface(
                        hfile["x"], times, np.array([hfile[k] for k in keys]), out
                    )
                )
    if not any(e["name"] == "thickness_history_3d" for e in entries):
        from .viz3d import thickness_profile_3d

        entries.append(thickness_profile_3d(state["h"], state["x"], out))
    return write_gallery(
        out,
        "Fixed-strip deposition: numerical evidence",
        "Spatial profiles, boundary-layer detail, field history and a complete material balance.",
        entries,
        [
            "This is the legacy floating-strip approximation, not the directly driven wire experiment.",
            "The parameters are uncalibrated. Numerically converged output is not experimental validation.",
            "The 3D history uses time as an axis; it does not reconstruct a physical volume.",
        ],
        [
            ("Full final state", "../final_state.npz"),
            ("Timeseries", "../timeseries.csv"),
            ("Every-step audit", "../integration_diagnostics.csv"),
            ("Scientific audit", "../research_audit.json"),
            ("Parameters", "../params.json"),
        ],
    )


def visualize_run(run_dir):
    """Re-render archived values without rerunning or changing the simulation."""
    run_dir = Path(run_dir)
    if (run_dir / "morphology.npz").exists():
        from .dendrites import load_dendrite_config

        with np.load(run_dir / "morphology.npz") as archive:
            result = dict(archive)
        result["records"] = pd.read_csv(run_dir / "timeseries.csv").to_dict(orient="records")
        result["metrics"] = json.loads((run_dir / "metrics.json").read_text())
        return dendrite_figures(result, load_dendrite_config(run_dir / "params.json"), run_dir)
    if (run_dir / "final_state.npz").exists():
        return strip_figures(run_dir)
    raise ValueError("Run directory has no supported archived final state")


def ensemble_figures(frame, bridging, conditions, root):
    """Show observations, not smoothed response surfaces across unlike conditions."""
    root = Path(root)
    out = root / "views"
    out.mkdir(exist_ok=True)
    entries = []
    groups = list(frame.groupby(conditions, dropna=False))
    with scientific_style():
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), layout="constrained")
        labels = []
        for index, (_, group) in enumerate(groups):
            labels.append(f"C{index + 1}\nT={group.period_steps.iloc[0]:g}")
            jitter = np.linspace(-0.12, 0.12, len(group))
            for ax, key in zip(axes[:2], ["deposited_cells", "area_asymmetry"]):
                values = group[key].to_numpy(float)
                keep = np.isfinite(values)
                ax.scatter(index + jitter[keep], values[keep], s=25, alpha=0.75, color=BLUE)
                if keep.any():
                    ax.plot(
                        [index - 0.18, index + 0.18],
                        [np.mean(values[keep])] * 2,
                        color=ORANGE,
                        lw=2,
                    )
            row = bridging.iloc[index]
            axes[2].errorbar(
                index,
                row.bridge_probability,
                yerr=[
                    [max(0, row.bridge_probability - row.bridge_probability_95_low)],
                    [max(0, row.bridge_probability_95_high - row.bridge_probability)],
                ],
                fmt="o",
                color=TEAL,
                capsize=4,
            )
        for ax in axes:
            ax.set_xticks(range(len(groups)), labels, rotation=45 if len(groups) > 5 else 0)
            ax.grid(axis="y")
        axes[0].set(ylabel="Deposited lattice cells", title="Each seed and its condition mean")
        axes[1].set(
            ylabel="Signed projected area imbalance",
            title="Asymmetry across seeds",
            ylim=(-1.05, 1.05),
        )
        axes[2].set(
            ylabel="Probability by planned horizon",
            title="Bridging: 95% Wilson intervals",
            ylim=(-0.03, 1.03),
        )
        entries.append(
            save_figure(
                fig,
                out,
                "ensemble_outcomes",
                "Which effects persist across independent seeds?",
                "Dots are individual model runs; orange bars are condition means. Bridge intervals are binomial model uncertainty. Condition IDs include all settings, not just period T. No experimental confidence is implied.",
                "independent model ensemble",
            )
        )
    table = []
    for index, (_, group) in enumerate(groups):
        table.append(
            {
                "condition": f"C{index + 1}",
                **{c: group[c].iloc[0] for c in conditions},
                "n": len(group),
            }
        )
    pd.DataFrame(table).to_csv(out / "conditions.csv", index=False)
    return write_gallery(
        out,
        "Direct-wire ensemble: repeatability",
        "Every replicate stays visible; distinct conditions are kept separate.",
        entries,
        [
            "Random-seed variability is model variability, not measurement or parameter uncertainty.",
            "Unbridged runs are censored at their planned horizon; their stopping times are not successful bridge times.",
            "Periods are model steps, not laboratory hertz. Condition IDs map to the full configuration table.",
        ],
        [
            ("All runs", "../results.csv"),
            ("Condition definitions", "conditions.csv"),
            ("Bridging and restricted-time estimates", "../bridging_summary.csv"),
        ],
    )


def manufacturing_figures(frame, out):
    from .manufacturing import integrate_signed_product

    out = Path(out)
    root = out / "views"
    root.mkdir(exist_ok=True)
    t, v, i = (frame[c].to_numpy(float) for c in ["time_s", "voltage_V", "current_A"])
    increments = np.array(
        [
            integrate_signed_product(t[k : k + 2], v[k : k + 2], i[k : k + 2])
            for k in range(len(t) - 1)
        ]
    )
    energy = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
    pd.DataFrame(
        {
            "time_s": t,
            "net_energy_J": energy[:, 0],
            "imported_energy_J": energy[:, 1],
            "returned_energy_J": energy[:, 2],
        }
    ).to_csv(out / "cumulative_electrical_energy.csv", index=False)
    with scientific_style():
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        axes[0, 0].plot(t, v, color=BLUE)
        axes[0, 0].set(ylabel="Terminal voltage [V]", title="Recorded drive")
        axes[0, 1].plot(t, i * 1e3, color=ORANGE)
        axes[0, 1].set(ylabel="Terminal current [mA]", title="Recorded response")
        axes[1, 0].plot(t, v * i, color=TEAL, marker=".", ms=3)
        axes[1, 0].set(ylabel="Sampled terminal power [W]", title="Power at recorded samples")
        axes[1, 1].plot(t, energy[:, 1], label="Imported", color=ORANGE)
        axes[1, 1].plot(t, energy[:, 2], label="Returned", color=TEAL)
        axes[1, 1].plot(t, energy[:, 0], label="Net", color=BLUE)
        axes[1, 1].legend(frameon=False)
        axes[1, 1].set(
            ylabel="Terminal electrical energy [J]", title="Exact piecewise-linear V × I integral"
        )
        for ax in axes.ravel():
            ax.set_xlabel("Time [s]")
            ax.grid(axis="y")
        entry = save_figure(
            fig,
            root,
            "electrical_accounting",
            "How much terminal electrical energy was imported and returned?",
            "Voltage/current lines connect recorded samples. Sampled power is not assumed linear for integration. Cumulative energy exactly integrates the product of the two linearly interpolated signals, including zero crossings.",
            "supplied measurement trace",
        )
    return write_gallery(
        root,
        "Measured terminal electrical accounts",
        "Traceable electrical measurements with explicit integration assumptions.",
        [entry],
        [
            "Instrument uncertainty is not supplied automatically and is not represented by a fabricated confidence band.",
            "Terminal current includes charging and side reactions; it is not automatically polymerization current.",
            "Electrical energy excludes auxiliary processing, supply losses and lifecycle impacts.",
        ],
        [
            ("Accounting results and input hashes", "../manufacturing_analysis.json"),
            ("Cumulative energy", "../cumulative_electrical_energy.csv"),
        ],
    )
