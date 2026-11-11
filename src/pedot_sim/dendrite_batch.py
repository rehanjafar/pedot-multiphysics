"""Reproducible direct-wire ensembles and censor-aware descriptive statistics."""
from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .dendrites import run_dendrite, analyze_dendrite_campaign
from .utils_io import safe_output_path, save_json, runtime_provenance


def summarize_bridging(frame, conditions):
    """Wilson intervals for model bridge probability and restricted mean time.

    Unbridged runs must reach the common planned horizon. With administrative
    censoring only, E[min(T,H)] is estimated by the ordinary mean of stopped
    times; it is not the mean time to bridge. Failed or interrupted runs cannot
    be treated as unbridged observations.
    """
    rows = []
    for values, group in frame.groupby(conditions, dropna=False):
        if len(conditions) == 1:
            values = (values,) if not isinstance(values, tuple) else values
        horizon = float(group.steps.iloc[0])
        elapsed = group.elapsed_model_steps.to_numpy(float)
        bridged = group.bridged.to_numpy()
        if not all(type(v) in (bool, np.bool_) for v in bridged):
            raise ValueError("Bridge outcomes must be boolean")
        if (not np.isfinite(elapsed).all() or np.any(elapsed <= 0) or np.any(elapsed > horizon)
                or not np.allclose(elapsed[~bridged], horizon, rtol=0, atol=1e-10)):
            raise ValueError("Unbridged observations must reach their planned horizon")
        n, k = len(group), int(bridged.sum())
        interval = binomtest(k, n).proportion_ci(confidence_level=.95, method="wilson")
        rows.append({**dict(zip(conditions, values)), "replicates": n,
                     "bridged_count": k, "censored_count": n-k,
                     "bridge_probability": k/n,
                     "bridge_probability_95_low": float(interval.low),
                     "bridge_probability_95_high": float(interval.high),
                     "restricted_mean_time_model_steps": float(elapsed.mean()),
                     "restricted_mean_standard_error": float(elapsed.std(ddof=1)/np.sqrt(n)) if n>1 else None,
                     "conditional_mean_bridge_time_model_steps": float(elapsed[bridged].mean()) if k else None})
    return pd.DataFrame(rows)


def run_dendrite_batch(config, campaign_id, output_base, periods, replicates=5):
    """Run a deterministic period/independent-seed plan, safely resumable.

    Serial execution keeps the exploratory sparse solver's memory use bounded.
    Completed runs are reused only if parameters, source fingerprint, environment
    and expected outputs match. Interrupted/failed run directories are retained
    and require a new campaign identifier instead of silently overwriting data.
    """
    config.validate()
    if type(replicates) is not int or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    periods = [float(p) for p in periods]
    if not periods or not np.isfinite(periods).all() or min(periods) <= 0 or len(set(periods)) != len(periods):
        raise ValueError("Periods must be unique positive finite values in model steps")
    if len(periods)*replicates > 10000:
        raise ValueError("Campaign exceeds 10000 runs")
    jobs = [(f"period{index:03d}_rep{rep:04d}", replace(config, period_steps=period,
             seed=config.seed+index*replicates+rep))
            for index, period in enumerate(periods) for rep in range(replicates)]
    for _, cfg in jobs:
        cfg.validate()
    root = safe_output_path(output_base, "dendrites", campaign_id)
    root.mkdir(parents=True, exist_ok=True)
    lock = root/"campaign.lock"
    with lock.open("x"):
        pass
    try:
        plan = {"version":1,"provenance":runtime_provenance(),
                "runs":{key:asdict(cfg) for key,cfg in jobs}}
        plan_path = root/"plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError("Existing campaign differs in parameters, code or environment; use a new campaign ID")
        else:
            if any(item.is_dir() for item in root.iterdir()):
                raise ValueError("Cannot adopt existing runs into an unrecorded campaign")
            save_json(plan, plan_path)
        completed = []
        for key, cfg in jobs:
            run_dir = safe_output_path(output_base, "dendrites", campaign_id, key)
            if run_dir.exists():
                required = ["metadata.json","params.json","metrics.json","morphology.npz","timeseries.csv","morphology.png"]
                if not all((run_dir/name).is_file() for name in required):
                    raise ValueError(f"Incomplete run {key}; preserved for inspection, use a new campaign ID")
                metadata = json.loads((run_dir/"metadata.json").read_text())
                if (metadata.get("status") != "completed" or metadata.get("provenance") != plan["provenance"]
                        or json.loads((run_dir/"params.json").read_text()) != asdict(cfg)):
                    raise ValueError(f"Cannot resume mismatched/failed run {key}; use a new campaign ID")
            else:
                run_dendrite(cfg, key, output_base, campaign_id)
            completed.append(key)
            save_json({"completed":completed, "planned":len(jobs)}, root/"manifest.json")
        return analyze_dendrite_campaign(campaign_id, output_base)
    finally:
        lock.unlink()
