"""Traceable subsets of the CC BY 4.0 Janzakova et al. (2021) source data.

Only experimental traces and image-derived observations are selected. The
source also contains circuit simulations; those are deliberately not included
in the measurement tables. Row and column indices below are zero based in
code and one based in exported provenance. No digitization or fitted values.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from importlib.resources import files
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .figure_style import (
    COLORS,
    BLUE,
    ORANGE,
    scientific_style,
    save_figure,
    save_interactive,
    write_gallery,
)
from .utils_io import save_json

SOURCE_SHA256 = "3a6dac6f58120cf5c05469952846da6718f72ddaac8cbc049d5cbec1070129fc"
SOURCE_DOI = "10.6084/m9.figshare.16814710.v1"
PAPER_DOI = "10.1038/s41467-021-27274-9"
TABLES = (
    "amplitude_morphology",
    "frequency_morphology",
    "offset_morphology",
    "growth_traces",
    "impedance",
    "cyclic_voltammetry",
)
AUTHORS = [
    "Kamila Janzakova",
    "Ankush Kumar",
    "Mahdi Ghazal",
    "Anna Susloparova",
    "Yannick Coffinier",
    "Fabien Alibart",
    "Sébastien Pecqueur",
]


def curate_source(source_csv, destination):
    """Recreate bundled tables from the exact versioned original CSV.

    Hash verification precedes parsing or writes; unexpected revisions require
    explicit scientific review rather than silently reusing positional offsets.
    """
    payload = Path(source_csv).read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("Source SHA-256 does not match the reviewed dataset version")
    rows = list(csv.reader(io.StringIO(payload.decode("cp1252")), delimiter=";"))
    tables = {}
    for name, start, stop, key in [
        ("amplitude_morphology", 794, 801, "amplitude_V"),
        ("frequency_morphology", 2458, 2470, "frequency_Hz"),
    ]:
        records = []
        for r in range(start, stop):
            vals = list(map(float, rows[r][:12]))
            records.append(
                {
                    key: vals[0],
                    "projected_area_m2": vals[1] * 1e-6,
                    "completion_time_s": vals[2],
                    "image_fractal_number": vals[3],
                    **{f"branching_position_{j + 1}": vals[4 + j] for j in range(6)},
                    "branching_mean": vals[10],
                    "branching_spatial_sd": vals[11],
                    "quality_status": "unresolved_zero_area" if vals[1] == 0 else "as_reported",
                    "source_row_1based": r + 1,
                }
            )
        tables[name] = pd.DataFrame(records)
    records = []
    for r in range(889, 899):
        v = list(map(float, rows[r][:6]))
        records.append(
            dict(
                offset_V=v[0],
                driven_area_m2=v[1] * 1e-6,
                grounded_area_m2=v[2] * 1e-6,
                completion_time_s=v[3],
                driven_branching=v[4],
                grounded_branching=v[5],
                source_row_1based=r + 1,
            )
        )
    tables["offset_morphology"] = pd.DataFrame(records)
    records = []
    for col, amplitude in enumerate(np.arange(3.5, 7.0, 0.5), start=1):
        for r in range(645, 792):
            if rows[r][col].strip():
                records.append(
                    dict(
                        amplitude_V=amplitude,
                        time_s=float(rows[r][0]),
                        projected_area_m2=float(rows[r][col]) * 1e-6,
                        source_row_1based=r + 1,
                        source_area_column_1based=col + 1,
                    )
                )
    tables["growth_traces"] = pd.DataFrame(records)
    records = []
    for sample in range(3):
        col = sample * 3
        for r in range(2472, 2533):
            f, modulus, phase = map(float, rows[r][col : col + 3])
            records.append(
                dict(
                    sample_id=sample + 1,
                    frequency_Hz=f,
                    impedance_modulus_ohm=modulus,
                    impedance_phase_deg=phase,
                    source_row_1based=r + 1,
                    source_frequency_column_1based=col + 1,
                )
            )
    tables["impedance"] = pd.DataFrame(records)
    records = []
    for col, chemistry in [(0, "NaPSS"), (20, "BQ + NaPSS"), (40, "EDOT + NaPSS")]:
        for r in range(3, 643):
            if rows[r][col].strip() and rows[r][col + 1].strip():
                records.append(
                    dict(
                        chemistry=chemistry,
                        cycle=1,
                        potential_V_vs_AgAgCl=float(rows[r][col]),
                        current_A=float(rows[r][col + 1]),
                        acquisition_index=r - 3,
                        source_row_1based=r + 1,
                        source_potential_column_1based=col + 1,
                    )
                )
    tables["cyclic_voltammetry"] = pd.DataFrame(records)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(destination / f"{name}.csv", index=False)
    metadata = {
        "authors": AUTHORS,
        "paper_doi": PAPER_DOI,
        "dataset_doi": SOURCE_DOI,
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_url": "https://ndownloader.figshare.com/files/31090411",
        "source_filename": "dataset.csv",
        "source_sha256": SOURCE_SHA256,
        "source_md5": "aafa210eee5ddfa4f44a48cf656623c1",
        "source_bytes": len(payload),
        "changes": [
            "Selected six experimental/image-analysis sections; excluded circuit modeling and fits.",
            "Reshaped to tidy CSV; converted mm² to m²; retained original row/column coordinates.",
            "Selected only first CV cycles; retained acquisition order and Ag/AgCl potential reference.",
            "Flagged the unresolved zero-area entry at 10 Hz; did not alter or impute it.",
        ],
        "sections": {
            name: {
                "rows": len(frame),
                "sha256": hashlib.sha256((destination / f"{name}.csv").read_bytes()).hexdigest(),
            }
            for name, frame in tables.items()
        },
        "experimental_context": {
            "gold_wire_diameter_m": 25e-6,
            "gap_m": 240e-6,
            "aqueous_droplet_m3": 20e-9,
            "EDOT_mol_m3": 10.0,
            "BQ_mol_m3": 10.0,
            "NaPSS_mol_m3": 1.0,
            "amplitude_series": "80 Hz, 50% duty, zero offset",
            "frequency_series": "5 V peak, 50% duty, zero offset",
            "offset_series": "5 V peak, 80 Hz, 50% duty",
            "impedance_medium": "1 mM NaPSS without EDOT or BQ; after growth",
            "concentration_basis": "NaPSS molarity as reported by the source; not an inferred polymer-chain concentration",
        },
        "uncertainty": "Branching SD is spatial variation across six image positions, not independent experiments. Instrument/segmentation uncertainties are not supplied here.",
        "calibration_status": "Published reference observations; no calibration to this simulator or the user experiment.",
    }
    save_json(metadata, destination / "source_metadata.json")
    return metadata


def load_reference_tables():
    """Load and integrity-check packaged evidence, including installed wheels."""
    root = files("pedot_sim").joinpath("datasets", "janzakova2021")
    metadata = json.loads(root.joinpath("source_metadata.json").read_text(encoding="utf-8"))
    tables = {}
    for name in TABLES:
        payload = root.joinpath(name + ".csv").read_bytes()
        if hashlib.sha256(payload).hexdigest() != metadata["sections"][name]["sha256"]:
            raise ValueError(f"Bundled reference table integrity check failed: {name}")
        tables[name] = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
    return tables, metadata


def analyze_literature(output_dir):
    """Create an offline evidence atlas; never fit model units to lab data."""
    tables, metadata = load_reference_tables()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    source = files("pedot_sim").joinpath("datasets", "janzakova2021")
    for name in tables:
        (out / f"{name}.csv").write_bytes(source.joinpath(name + ".csv").read_bytes())
    (out / "ATTRIBUTION.md").write_bytes(source.joinpath("ATTRIBUTION.md").read_bytes())
    save_json(metadata, out / "source_metadata.json")
    entries = []
    origin = "published experiment · Janzakova et al. 2021"
    with scientific_style():
        for name, key, label, context in [
            (
                "amplitude_morphology",
                "amplitude_V",
                "Peak amplitude [V]",
                "80 Hz; 50% duty; zero offset",
            ),
            (
                "frequency_morphology",
                "frequency_Hz",
                "Drive frequency [Hz]",
                "5 V peak; 50% duty; zero offset",
            ),
        ]:
            df = tables[name]
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
            specs = [
                ("projected_area_m2", 1e6, "Projected area [mm²]", "Projected material footprint"),
                (
                    "completion_time_s",
                    1,
                    "Reported completion time [s]",
                    "Time to the reported endpoint",
                ),
                (
                    "image_fractal_number",
                    1,
                    "2D image fractal number",
                    "Image-scale branching complexity",
                ),
                ("branching_mean", 1, "Branching degree", "Mean ± spatial SD (six positions)"),
            ]
            for ax, (metric, scale, ylabel, title) in zip(axes.ravel(), specs):
                ax.scatter(df[key], df[metric] * scale, color=BLUE, s=32, zorder=3)
                if metric == "branching_mean":
                    ax.errorbar(
                        df[key],
                        df[metric],
                        yerr=df.branching_spatial_sd,
                        fmt="none",
                        color=BLUE,
                        capsize=3,
                    )
                ax.set(xlabel=label, ylabel=ylabel, title=title)
                ax.grid(axis="y")
                if key == "frequency_Hz":
                    ax.set_xscale("log")
            if key == "frequency_Hz":
                axes[0, 0].scatter([10], [0], marker="x", s=80, color=ORANGE, zorder=4)
                axes[0, 0].annotate(
                    "Reported zero: unresolved anomaly",
                    xy=(10, 0),
                    xytext=(18, 0.015),
                    fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": ORANGE},
                )
            fig.suptitle(context, fontsize=12, color="#466078")
            entries.append(
                save_figure(
                    fig,
                    out,
                    name,
                    "How do the reported morphologies vary with "
                    + ("amplitude?" if key == "amplitude_V" else "frequency?"),
                    "Direct source values; no fitted optimum or interpolation. Error bars describe six positions within each morphology, not replicate-to-replicate uncertainty. "
                    + (
                        "The 10 Hz area of zero is preserved and flagged; it conflicts with nonzero morphology measures."
                        if key == "frequency_Hz"
                        else ""
                    ),
                    origin,
                )
            )
        growth = tables["growth_traces"]
        fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
        fig3 = plt.figure(figsize=(9, 6.5), layout="constrained")
        ax3 = fig3.add_subplot(projection="3d")
        interactive = go.Figure()
        for color, (amplitude, df) in zip(COLORS, growth.groupby("amplitude_V")):
            x, y = df.time_s.to_numpy(), df.projected_area_m2.to_numpy() * 1e6
            ax.plot(x, y, color=color, label=f"{amplitude:g} V")
            ax3.plot(x, np.full(len(x), amplitude), y, color=color, lw=1.8)
            interactive.add_trace(
                go.Scatter3d(
                    x=x.tolist(),
                    y=[amplitude] * len(x),
                    z=y.tolist(),
                    mode="lines+markers",
                    marker={"size": 2},
                    line={"color": color, "width": 4},
                    name=f"{amplitude:g} V",
                )
            )
        ax.set(
            xlabel="Recorded growth time [s]",
            ylabel="Projected area [mm²]",
            title="Image-derived growth traces at 80 Hz",
        )
        ax.legend(title="Peak amplitude", ncol=2, frameon=False)
        ax.grid(axis="y")
        entries.append(
            save_figure(
                fig,
                out,
                "growth_traces",
                "How does projected area develop over time?",
                "Every reported nonblank sample is retained. Lines connect observations within one voltage series. Initial area is not subtracted; missing tails are not extrapolated.",
                origin,
            )
        )
        ax3.set(
            xlabel="Time [s]",
            ylabel="Peak amplitude [V]",
            zlabel="Projected area [mm²]",
            title="Measured growth in parameter space",
        )
        ax3.view_init(26, -60)
        entry = save_figure(
            fig3,
            out,
            "growth_parameter_3d",
            "How do measured growth histories compare in three coordinates?",
            "Three axes encode time, applied peak amplitude and image area. This is a parameter-space view of observations, not physical 3D morphology. There is no surface between voltage series.",
            origin,
        )
        entry["interactive"] = save_interactive(
            interactive,
            out,
            "growth_parameter_3d",
            "Published growth histories",
            "Measured time × applied amplitude × projected area; no physical 3D reconstruction.",
            {
                "xaxis_title": "Time [s]",
                "yaxis_title": "Peak amplitude [V]",
                "zaxis_title": "Projected area [mm²]",
            },
        )
        entries.append(entry)
        offset = tables["offset_morphology"]
        asymmetry = (offset.driven_area_m2 - offset.grounded_area_m2) / (
            offset.driven_area_m2 + offset.grounded_area_m2
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
        axes[0].scatter(offset.offset_V, offset.driven_area_m2 * 1e6, label="Driven", color=ORANGE)
        axes[0].scatter(
            offset.offset_V, offset.grounded_area_m2 * 1e6, label="Grounded", color=BLUE
        )
        axes[0].set(ylabel="Projected area [mm²]", title="Area on each connected wire")
        axes[0].legend(frameon=False)
        axes[1].scatter(offset.offset_V, asymmetry, color=BLUE)
        axes[1].axhline(0, lw=1, color="#9AABB8")
        axes[1].set(
            ylabel="(driven − grounded) / total",
            ylim=(-1.05, 1.05),
            title="Area imbalance derived from the observations",
        )
        for ax in axes:
            ax.set_xlabel("Applied DC offset [V]")
            ax.grid(axis="y")
        entries.append(
            save_figure(
                fig,
                out,
                "offset_asymmetry",
                "How does drive offset redistribute the projected material?",
                "Areas are source observations; signed imbalance is computed from those two areas. No replicate uncertainty was supplied. Conditions: 5 V peak, 80 Hz, 50% duty.",
                origin,
            )
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
        derived = []
        for sample, df in tables["impedance"].groupby("sample_id"):
            real = df.impedance_modulus_ohm * np.cos(np.deg2rad(df.impedance_phase_deg))
            minus_imag = -df.impedance_modulus_ohm * np.sin(np.deg2rad(df.impedance_phase_deg))
            axes[0].loglog(df.frequency_Hz, df.impedance_modulus_ohm, label=f"Sample {sample}")
            axes[1].semilogx(df.frequency_Hz, df.impedance_phase_deg)
            axes[2].plot(real / 1e3, minus_imag / 1e3, marker=".", ms=3)
            derived.append(
                df.assign(real_impedance_ohm=real, negative_imaginary_impedance_ohm=minus_imag)
            )
        pd.concat(derived).to_csv(out / "impedance_cartesian.csv", index=False)
        axes[0].set(xlabel="Probe frequency [Hz]", ylabel="|Z| [Ω]", title="Impedance magnitude")
        axes[0].legend(frameon=False)
        axes[1].set(
            xlabel="Probe frequency [Hz]", ylabel="Phase [degrees]", title="Impedance phase"
        )
        axes[2].set(xlabel="Z′ [kΩ]", ylabel="−Z″ [kΩ]", title="Cartesian representation")
        axes[2].set_aspect("equal", adjustable="datalim")
        entries.append(
            save_figure(
                fig,
                out,
                "impedance",
                "How do the three recorded samples respond electrically?",
                "Fig. 4c raw impedance, measured after growth in 1 mM NaPSS without EDOT or BQ. Sample IDs are retained without inferring morphology identities. Z′ = |Z| cos(phase); −Z″ = −|Z| sin(phase). Probe frequency is not growth-drive frequency. No equivalent-circuit fit.",
                origin,
            )
        )
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout="constrained", sharex=True)
        for ax, (chemistry, df) in zip(
            axes, tables["cyclic_voltammetry"].groupby("chemistry", sort=False)
        ):
            ax.plot(df.potential_V_vs_AgAgCl, df.current_A * 1e6, lw=1.2)
            ax.set(
                xlabel="Potential [V vs Ag/AgCl]",
                ylabel="Measured current [µA]",
                title=chemistry + " · cycle 1",
            )
            ax.grid(axis="y")
        entries.append(
            save_figure(
                fig,
                out,
                "cyclic_voltammetry",
                "Which electrochemical signals are observed in the separate solutions?",
                "First cycles from Fig. 2a, connected in original acquisition order. Separate current scales preserve detail. Reference-electrode potentials must not be equated with the two-wire growth voltage.",
                origin,
            )
        )
    notes = [
        "Published reference observations, not a calibration of this simulator or your experimental apparatus.",
        "Branching error bars are spatial standard deviations. Neither instrument uncertainty nor growth repeatability is inferred.",
        "The reported 10 Hz projected area is zero despite nonzero morphology measures. It remains unresolved and is not used for a fitted trend.",
        "Three-dimensional plots encode parameters; 2D optical areas do not establish physical volume, dry mass or 3D fractal dimension.",
        "Data: Janzakova and coauthors (2021), CC BY 4.0. Units and layout were transformed as documented; no numerical values were imputed.",
    ]
    return write_gallery(
        out,
        "Published PEDOT evidence: directly driven gold wires",
        "Traceable experimental and image-derived observations, curated from the authors’ numerical source data.",
        entries,
        notes,
        [(name.replace("_", " ").title(), name + ".csv") for name in TABLES]
        + [
            ("Source, license and transformations", "source_metadata.json"),
            ("Original dataset", "https://doi.org/" + SOURCE_DOI),
            ("Research paper", "https://doi.org/" + PAPER_DOI),
            ("Derived impedance coordinates", "impedance_cartesian.csv"),
        ],
    )
