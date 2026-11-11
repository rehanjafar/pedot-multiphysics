"""Directly driven gold-wire dendrites: exploratory charged-aggregate lattice model.

Motivated by Kumar et al., Sci. Rep. 12, 6395 (2022), DOI
10.1038/s41598-022-10082-6. This is an independent implementation, not a
reproduction of their code or a calibration to their experiments. A walker
represents one aggregate occupying one lattice cell. Space is in cells and
elapsed time in model steps; neither is mapped to monomers or seconds.

The deposited network inherits the potential of its connected wire. Laplace's
equation is re-solved after every geometry update. Positive local metal-minus-
solution voltage enables attachment. There is no explicit counter-reaction or
interfacial impedance; charge conservation of electrochemistry is not claimed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse.linalg import spsolve

from .transport import _diffusion_operator
from .utils_io import safe_output_path, save_json, save_npz, save_csv, runtime_provenance
from .waveforms import pulsed


@dataclass
class DendriteConfig:
    nx: int = 61
    ny: int = 41
    wire_length: int = 10
    wire_width: int = 3
    n_particles: int = 500
    steps: int = 400
    period_steps: float = 40.
    amplitude_V: float = 5.
    offset_V: float = 0.
    duty_cycle: float = .5
    diffusion_cells2_per_step: float = .15
    mobility_cells2_per_V_step: float = .5
    species_charge_sign: int = 1
    attachment_probability: float = .25
    activation_voltage_V: float = .02
    seed: int = 42
    snapshot_stride: int = 20

    def validate(self):
        for name in ("nx", "ny", "wire_length", "wire_width", "n_particles", "steps", "snapshot_stride"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if min(self.nx, self.ny) < 7 or 2*self.wire_length+3 >= self.nx:
            raise ValueError("Grid must contain two separated wires and a resolved gap")
        if self.wire_width >= self.ny-2 or self.wire_width % 2 != 1:
            raise ValueError("wire_width must be odd and leave fluid above and below the wires")
        if self.n_particles > self.nx*self.ny-2*self.wire_length*self.wire_width:
            raise ValueError("Too many aggregates for available fluid cells")
        if self.nx*self.ny > 250_000 or self.steps > 1_000_000:
            raise ValueError("Requested lattice exceeds this exploratory solver's resource limits")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.species_charge_sign not in (-1, 0, 1):
            raise ValueError("species_charge_sign must be -1, 0 or 1")
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All dendrite parameters must be finite")
        if self.period_steps <= 0 or self.amplitude_V < 0 or not 0 < self.duty_cycle < 1:
            raise ValueError("Invalid waveform")
        if min(self.diffusion_cells2_per_step, self.mobility_cells2_per_V_step, self.activation_voltage_V) < 0:
            raise ValueError("Transport and activation parameters must be nonnegative")
        if not 0 <= self.attachment_probability <= 1:
            raise ValueError("attachment_probability must lie in [0,1]")


def initial_electrodes(config):
    """Labels 1/2 identify conducting networks attached to driven/ground wires."""
    owner = np.zeros((config.ny, config.nx), dtype=np.int8)
    start = config.ny//2-config.wire_width//2
    owner[start:start+config.wire_width, :config.wire_length] = 1
    owner[start:start+config.wire_width, -config.wire_length:] = 2
    return owner


def solve_network_potential(owner, applied_voltage):
    """Insulating box, ideal equipotential networks: left driven, right grounded."""
    mask = np.ones(owner.shape, bool)
    L = _diffusion_operator(SimpleNamespace(dx=1., dy=1.), mask)
    solid = owner.ravel() > 0
    free = ~solid
    potential = np.where(owner.ravel() == 1, applied_voltage, 0.).astype(float)
    potential[free] = spsolve(L[free][:, free], -L[free][:, solid]@potential[solid])
    if not np.isfinite(potential).all():
        raise ValueError("Nonfinite dendrite potential")
    residual = float(np.max(np.abs((L@potential)[free]))) if free.any() else 0.
    if residual > 1e-8*max(1., abs(applied_voltage)):
        raise ValueError("Dendrite Laplace residual exceeds tolerance")
    return potential.reshape(owner.shape), residual


def morphology_metrics(owner, initial, mobile_count, config, elapsed):
    deposited = (owner > 0) & (initial == 0)
    left = int(np.sum(deposited & (owner == 1)))
    right = int(np.sum(deposited & (owner == 2)))
    count = left+right
    if count+mobile_count != config.n_particles:
        raise ValueError("Aggregate inventory is not conserved")
    adjacent = ((owner[:, 1:]*owner[:, :-1] == 2).any()
                or (owner[1:, :]*owner[:-1, :] == 2).any())
    return {"deposited_cells": count, "left_deposited_cells": left,
            "right_deposited_cells": right, "mobile_aggregates": mobile_count,
            "area_asymmetry": (left-right)/count if count else None,
            "projected_coverage_fraction": count/int(np.sum(initial == 0)),
            "bridged": bool(adjacent), "elapsed_model_steps": elapsed,
            "deposited_cells_per_model_step": count/elapsed if elapsed else 0.}


def simulate_dendrites(config: DendriteConfig):
    """Stochastic, exclusion-lattice drift/diffusion with irreversible attachment.

    Hops use conservative upwind drift plus symmetric diffusion rates, subcycled
    so total move probability <=0.8. Failed hops reflect at walls, conductors and
    other aggregates. Attachment is tested once per model step. Stop at first
    bridge: an ideal conductor connecting unequal imposed potentials is invalid.
    """
    config.validate()
    rng = np.random.default_rng(config.seed)
    owner = initial_electrodes(config)
    initial = owner.copy()
    cells = rng.choice(np.flatnonzero(owner == 0), config.n_particles, replace=False)
    positions = np.column_stack(np.unravel_index(cells, owner.shape))
    alive = np.ones(config.n_particles, bool)
    occupied = np.zeros(owner.shape, bool)
    occupied[positions[:, 0], positions[:, 1]] = True
    snapshots, records = [owner.copy()], []
    unit, residual = solve_network_potential(owner, 1.)
    max_residual = residual
    dirty = False
    directions = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    for step in range(config.steps):
        if dirty:
            unit, residual = solve_network_potential(owner, 1.)
            max_residual = max(max_residual, residual)
        voltage = float(pulsed(step+.5, config.offset_V, config.amplitude_V,
                               1/config.period_steps, config.duty_cycle))
        V = unit*voltage
        # Face rates toward each neighbor; no particle flux across external walls.
        rates = np.zeros((*owner.shape, 4))
        for k, (dy, dx) in enumerate(directions):
            dest = np.roll(V, (-dy, -dx), (0, 1))
            drift = -config.species_charge_sign*config.mobility_cells2_per_V_step*(dest-V)
            rates[:, :, k] = config.diffusion_cells2_per_step+np.maximum(drift, 0.)
            if dx == 1: rates[:, -1, k] = 0
            if dx == -1: rates[:, 0, k] = 0
            if dy == 1: rates[-1, :, k] = 0
            if dy == -1: rates[0, :, k] = 0
        substeps = max(1, int(np.ceil(np.max(rates.sum(axis=2))/.8)))
        if substeps > 10000:
            raise ValueError("Transport rates require excessive subcycling; reduce mobility or voltage")
        for _ in range(substeps):
            for i in rng.permutation(np.flatnonzero(alive)):
                y, x = positions[i]
                k = int(np.searchsorted(np.cumsum(rates[y, x])/substeps, rng.random()))
                if k == 4:
                    continue
                yy, xx = positions[i]+directions[k]
                if 0 <= yy < config.ny and 0 <= xx < config.nx and owner[yy, xx] == 0 and not occupied[yy, xx]:
                    occupied[y, x] = False
                    occupied[yy, xx] = True
                    positions[i] = yy, xx
        # All attachment decisions use the pre-update geometry and potential.
        additions = []
        for i in rng.permutation(np.flatnonzero(alive)):
            y, x = positions[i]
            candidates = []
            for dy, dx in directions:
                yy, xx = y+dy, x+dx
                if 0 <= yy < config.ny and 0 <= xx < config.nx and owner[yy, xx]:
                    label = int(owner[yy, xx])
                    metal = voltage if label == 1 else 0.
                    if metal-V[y, x] > config.activation_voltage_V:
                        candidates.append((metal-V[y, x], label))
            if candidates and rng.random() < config.attachment_probability:
                additions.append((i, max(candidates)[1]))
        dirty = bool(additions)
        for i, label in additions:
            y, x = positions[i]
            owner[y, x] = label
            occupied[y, x] = False
            alive[i] = False
        metrics = morphology_metrics(owner, initial, int(alive.sum()), config, step+1)
        records.append({"step": step+1, "applied_voltage_V": voltage, **metrics})
        if (step+1) % config.snapshot_stride == 0 or metrics["bridged"]:
            snapshots.append(owner.copy())
        if metrics["bridged"]:
            break
    if not np.array_equal(snapshots[-1], owner):
        snapshots.append(owner.copy())
    # Do not solve a shorted geometry. The last solved potential is explicitly pre-update.
    return {"owner": owner, "initial_owner": initial, "mobile_positions": positions[alive],
            "snapshots": np.asarray(snapshots), "records": records, "metrics": metrics,
            "max_laplace_residual": max_residual}


def run_dendrite(config, run_id, output_base, campaign_id="dendrites"):
    """Save a reproducible morphology experiment, with units and calibration limits."""
    import pandas as pd
    config.validate()
    run_dir = safe_output_path(output_base, "dendrites", campaign_id, run_id)
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    save_json(asdict(config), run_dir/"params.json")
    try:
        result = simulate_dendrites(config)
        save_npz({key: result[key] for key in ("owner", "initial_owner", "mobile_positions", "snapshots")}, run_dir/"morphology.npz")
        save_csv(pd.DataFrame(result["records"]), run_dir/"timeseries.csv")
        save_json(result["metrics"], run_dir/"metrics.json")
        save_json({"provenance": runtime_provenance(), "status": "completed", "max_laplace_residual": result["max_laplace_residual"],
                   "spatial_unit": "lattice_cell", "time_unit": "model_step",
                   "experimental_validation": "not_established", "seed": config.seed,
                   "source": "https://doi.org/10.1038/s41598-022-10082-6",
                   "limitations": ["No mapping to physical length, time, concentration, yield or energy.",
                       "Ideal conducting deposits; no interfacial impedance or explicit counter-reaction.",
                       "Aggregate attachment is phenomenological; no molecular polymerization or dopant model.",
                       "Growth stops at first contact; no electrical solution is claimed after bridging."]}, run_dir/"metadata.json")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        ax.imshow(result["owner"], origin="lower", cmap=ListedColormap(["white", "#b35b19", "#225e9c"]), vmin=0, vmax=2)
        points = result["mobile_positions"]
        if len(points):
            ax.scatter(points[:, 1], points[:, 0], s=2, c="#aaaaaa", label="Mobile aggregates")
        ax.set(xlabel="x [lattice cells]", ylabel="y [lattice cells]",
               title="Directly driven wire growth — uncalibrated aggregate model")
        fig.savefig(run_dir/"morphology.png", dpi=180)
        plt.close(fig)
        return {"run_dir": str(run_dir), **result["metrics"]}
    except Exception as exc:
        save_json({"status": "failed", "error": str(exc)}, run_dir/"metadata.json")
        raise


def load_dendrite_config(path):
    config = DendriteConfig(**json.loads(Path(path).read_text()))
    config.validate()
    return config


def analyze_dendrite_campaign(campaign_id, output_base):
    """Aggregate like-for-like model replicates; never convert model steps to Hz."""
    import html
    import pandas as pd
    root = safe_output_path(output_base, "dendrites", campaign_id)
    rows = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/"metadata.json").exists():
            continue
        if json.loads((directory/"metadata.json").read_text()).get("status") != "completed":
            continue
        rows.append({"run_id": directory.name,
                     **json.loads((directory/"params.json").read_text()),
                     **json.loads((directory/"metrics.json").read_text())})
    if not rows:
        raise ValueError("No completed dendrite runs")
    df = pd.DataFrame(rows)
    save_csv(df, root/"results.csv")
    conditions = [key for key in asdict(DendriteConfig()) if key != "seed"]
    duplicated = df.duplicated(conditions+["seed"], keep=False)
    if duplicated.any():
        raise ValueError("Duplicate seed/condition runs are not independent replicates")
    outcomes = ["deposited_cells", "area_asymmetry", "projected_coverage_fraction", "elapsed_model_steps"]
    summary = df.groupby(conditions, dropna=False)[outcomes].agg(["count", "mean", "std", "min", "max"]).reset_index()
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    save_csv(summary, root/"descriptive_stats.csv")
    save_json({"status": "exploratory", "n_runs": len(df),
               "interpretation": "Dispersion across random seeds measures this model's stochastic variability, not experimental uncertainty.",
               "bridging": "Compare completion time only with the bridged flag: unbridged runs are right-censored at their stopping step.",
               "units": "Lattice cells, model steps and applied volts. No calibrated manufacturing recommendation."}, root/"analysis_decisions.json")
    report = ("<!doctype html><meta charset='utf-8'><title>Direct-wire dendrite campaign</title>"
              "<h1>Directly driven wire dendrites: "+html.escape(campaign_id)+"</h1>"
              "<p>Exploratory aggregate model. No experimental validation or physical length/time calibration. "
              "Seed-to-seed variability is model variability. Unbridged runs have censored completion times.</p>"
              "<h2>All outcomes and parameters</h2>"+df.to_html(index=False, escape=True)+
              "<h2>Like-for-like replicate summaries</h2>"+summary.to_html(index=False, escape=True))
    for run in df.run_id:
        report += '<figure><img width="650" src="'+html.escape(run, quote=True)+'/morphology.png"><figcaption>'+html.escape(run)+'</figcaption></figure>'
    (root/"report.html").write_text(report, encoding="utf-8")
    return root/"report.html"
