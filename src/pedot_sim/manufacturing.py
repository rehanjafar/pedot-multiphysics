"""Dimensioned manufacturing accounts from measured data, separate from simulators."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .utils_io import ensure_dir, save_json


def integrate_signed_product(time, voltage, current):
    """Exact integral of the product of piecewise-linear sampled signals.

    Splits intervals at each voltage/current zero crossing before separating
    imported and returned energy. This cannot recover unsampled fast pulses.
    Returns net, positive and negative-magnitude integrals.
    """
    t, v, i = (np.asarray(x, dtype=float) for x in (time, voltage, current))
    if any(x.ndim != 1 for x in (t, v, i)) or not len(t)==len(v)==len(i) or len(t)<2:
        raise ValueError("Signals must be matching one-dimensional arrays with >=2 samples")
    if not all(np.isfinite(x).all() for x in (t,v,i)) or np.any(np.diff(t)<=0):
        raise ValueError("Signals must be finite and timestamps strictly increasing")
    positive = negative = 0.
    for dt, v0, v1, i0, i1 in zip(np.diff(t),v[:-1],v[1:],i[:-1],i[1:]):
        dv, di = v1-v0, i1-i0
        cuts = [0.,1.]
        for a, b in ((v0,dv),(i0,di)):
            if b and 0 < -a/b < 1:
                cuts.append(-a/b)
        cuts = sorted(set(cuts))
        def primitive(s):
            return v0*i0*s+(v0*di+i0*dv)*s*s/2+dv*di*s**3/3
        for a,b in zip(cuts[:-1],cuts[1:]):
            amount = dt*(primitive(b)-primitive(a))
            if (v0+dv*(a+b)/2)*(i0+di*(a+b)/2) >= 0:
                positive += amount
            else:
                negative -= amount
    return float(positive-negative), float(positive), float(negative)


def analyze_manufacturing(trace_path, metadata_path, output_dir):
    """Measured terminal energy/charge, yield-normalized energy and film properties.

    Inputs carry their own provenance. No current, conductivity, thickness, mass,
    yield, or lifecycle savings are inferred from the uncalibrated growth models.
    """
    trace_path, metadata_path = Path(trace_path), Path(metadata_path)
    frame = pd.read_csv(trace_path)
    metadata = json.loads(metadata_path.read_text())
    required = {"time_s", "voltage_V", "current_A"}
    if not required <= set(frame):
        raise ValueError(f"Measured trace requires columns {sorted(required)}")
    if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
        raise ValueError("Measurement metadata requires a source identifier")
    if metadata.get("units") != {"time_s":"s","voltage_V":"V","current_A":"A"}:
        raise ValueError("Trace units must explicitly be seconds, volts and amperes")
    t,v,i = [frame[col].to_numpy(float) for col in ("time_s","voltage_V","current_A")]
    net, imported, returned = integrate_signed_product(t,v,i)
    charge, positive, negative = integrate_signed_product(t,np.ones_like(t),i)
    result = {"duration_s":float(t[-1]-t[0]), "net_terminal_energy_J":net,
              "imported_terminal_energy_J":imported,"returned_terminal_energy_J":returned,
              "net_charge_C":charge,"positive_charge_C":positive,"negative_charge_C":negative,
              "mean_terminal_power_W":net/(t[-1]-t[0]), "peak_abs_voltage_V":float(np.max(np.abs(v))),
              "peak_abs_current_A":float(np.max(np.abs(i)))}
    warnings = ["Terminal electrical energy excludes auxiliary processing and lifecycle impacts.",
                "Charge includes every measured process, including charging and side reactions; it is not automatically polymerization charge.",
                "Integration assumes signals vary linearly between recorded samples."]
    values = {}
    for name in ("dry_product_mass_kg","film_area_m2","mean_thickness_m","four_terminal_resistance_ohm",
                 "voltage_probe_spacing_m","conducting_width_m","initial_monomer_mass_kg","frequency_Hz"):
        if name in metadata:
            value = metadata[name]
            if type(value) not in (int,float) or not np.isfinite(value) or value<=0:
                raise ValueError(f"{name} must be positive and finite")
            values[name] = float(value)
    mass = values.get("dry_product_mass_kg")
    if mass:
        result["imported_specific_energy_J_per_kg"] = imported/mass
        result["net_specific_energy_J_per_kg"] = net/mass
        if "initial_monomer_mass_kg" in values:
            result["product_to_initial_monomer_mass_ratio"] = mass/values["initial_monomer_mass_kg"]
            warnings.append("Product/monomer mass ratio is not chemical conversion; dopants, losses and solvent retention change product mass.")
        if {"film_area_m2","mean_thickness_m"} <= values.keys():
            result["effective_dry_density_kg_per_m3"] = mass/(values["film_area_m2"]*values["mean_thickness_m"])
    resistance = values.get("four_terminal_resistance_ohm")
    if resistance and {"voltage_probe_spacing_m","conducting_width_m"} <= values.keys():
        sheet = resistance*values["conducting_width_m"]/values["voltage_probe_spacing_m"]
        result["effective_sheet_resistance_ohm_per_square"] = sheet
        if "mean_thickness_m" in values:
            result["effective_conductivity_S_per_m"] = 1/(sheet*values["mean_thickness_m"])
        warnings.append("Resistance conversion assumes a uniform rectangular conducting path between voltage probes; branched networks require a geometry-specific electrical model.")
    if "frequency_Hz" in values:
        samples = 1/(values["frequency_Hz"]*np.max(np.diff(t)))
        result["minimum_samples_per_cycle"] = float(samples)
        if samples<40:
            warnings.append("Fewer than 40 samples per cycle; fast waveform features may be unresolved.")
    report={"status":"analyzed_measurements","metrics":result,"input_metadata":metadata,
            "warnings":warnings,"uncertainty":"Not propagated; input/instrument uncertainty must accompany physical conclusions.",
            "sha256":{key:hashlib.sha256(path.read_bytes()).hexdigest()
                      for key,path in (("trace",trace_path),("metadata",metadata_path))}}
    out=ensure_dir(Path(output_dir));save_json(report,out/"manufacturing_analysis.json")
    pd.DataFrame([result]).to_csv(out/"manufacturing_metrics.csv",index=False)
    import html
    (out/"manufacturing_report.html").write_text(
        "<!doctype html><meta charset='utf-8'><h1>Measured manufacturing accounts</h1><pre>"+
        html.escape(json.dumps(report,indent=2))+"</pre>",encoding="utf-8")
    from .research_figures import manufacturing_figures
    manufacturing_figures(frame, out)
    return report
