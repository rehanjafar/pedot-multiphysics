# pedot-multiphysics

Staged multiphysics simulation of waveform-controlled PEDOT electropolymerization in a two-compartment salt-bridge bipolar electrochemical system.

## Overview

This package models how waveform parameters (frequency, amplitude, DC offset), electrolyte properties (diffusion coefficient / viscosity), and cell geometry (bridge width) influence PEDOT film morphology on a floating bipolar electrode (BPE). The simulation couples four physics stages of increasing complexity:

| Stage | Physics | PDE |
|-------|---------|-----|
| **A** | Diffusion-reaction | dC/dt = D nabla^2 C - R |
| **B** | + Laplace electric field | nabla^2 V = 0, field-enhanced R |
| **C** | + Nernst-Planck migration | ion flux includes zFDC/RT nabla V |
| **D** | + Butler-Volmer kinetics | j = j0 (C/Cref)[exp(alpha nF eta/RT) - exp(-(1-alpha)nF eta/RT)] |

**Key dependent variables:**
- **Anisotropy Index (AI)** = h_max / h_mean (1.0 = perfectly uniform)
- **Uniformity Index (UI)** = std(h) (0.0 = perfectly uniform)

**Central hypothesis:** Uniformity (low AI) is achieved when tau_diff << tau_rxn (high frequency, low viscosity). Anisotropy (high AI) emerges when tau_diff > tau_rxn.

## Installation

```bash
# Clone and install in editable mode
cd pedot-multiphysics
pip install -e .

# With optional 3D visualization
pip install -e ".[viz3d]"
```

**Requirements:** Python 3.9+, NumPy, SciPy, Matplotlib, Pandas, statsmodels, Jinja2.

## Quick Start

### Single run
```bash
pedot-sim run-single \
  --config configs/example_verify.json \
  --run-id smoke_001 \
  --campaign-id quick_test \
  -v
```

### Full factorial campaign (54 runs)
```bash
pedot-sim run-batch \
  --config configs/example_factorial.json \
  --campaign-id factorial_v1 \
  --max-workers 4 \
  --sweep factorial \
  -v
```

### Dense parameter sweep (500+ runs)
```bash
pedot-sim run-batch \
  --config configs/example_dense.json \
  --campaign-id dense_v1 \
  --sweep dense \
  --max-workers 8 \
  -v
```

### Numerical verification
```bash
pedot-sim verify \
  --config configs/example_verify.json \
  --campaign-id verify_v1 \
  -v
```

### Statistical analysis
```bash
pedot-sim analyze --campaign-id factorial_v1 --output-base data
```

### Report generation
```bash
pedot-sim make-report --campaign-id factorial_v1 --output-base data
```

## Project Structure

```
pedot-multiphysics/
├── src/pedot_sim/
│   ├── __init__.py          # Package init
│   ├── config.py            # Dataclass-based configuration (SimConfig)
│   ├── constants.py         # Physical constants (F, R, D_LEVELS, etc.)
│   ├── geometry.py          # Two-compartment salt-bridge domain builder
│   ├── waveforms.py         # V(t) = V_DC + V_AC*sin(2*pi*f*t) + pulsed, ADC
│   ├── field.py             # Laplace solver (SOR) + electric field computation
│   ├── transport.py         # Diffusion (semi-implicit) + Nernst-Planck migration
│   ├── kinetics.py          # Butler-Volmer electrode kinetics
│   ├── nucleation.py        # Stochastic nucleation model (seeded RNG)
│   ├── growth.py            # Master timestep orchestrator (evolve_one_timestep)
│   ├── metrics.py           # AI, UI, depletion depth, timescale ratio, roughness
│   ├── runner_single.py     # Single-run executor with output serialization
│   ├── runner_batch.py      # Parallel batch campaigns (factorial + dense sweeps)
│   ├── viz2d.py             # 2D visualization (heatmaps, cross-sections, timeseries)
│   ├── viz3d.py             # 3D PyVista surface plots and animations
│   ├── analysis_stats.py    # 3-way ANOVA, polynomial regression, campaign plots
│   ├── analysis_opt.py      # Pareto front optimization
│   ├── verification.py      # Mesh + timestep refinement (Richardson extrapolation)
│   ├── validation.py        # Experimental data comparison scaffold
│   ├── reporting.py         # Jinja2 HTML + PDF report generator
│   ├── cli.py               # CLI entry point (pedot-sim)
│   ├── utils_io.py          # I/O utilities, RunManifest
│   ├── utils_grid.py        # Grid2D dataclass
│   └── logging_setup.py     # Logging configuration
├── configs/
│   ├── example_factorial.json   # 3x3x2x3 = 54 runs
│   ├── example_dense.json       # 500-point LHS sweep
│   └── example_verify.json      # Quick verification config
├── tests/
│   ├── test_metrics.py      # Unit tests for metrics
│   ├── test_waveforms.py    # Unit tests for waveforms
│   └── test_smoke_run.py    # End-to-end smoke tests (all 4 stages)
├── reports/templates/        # Jinja2 HTML + Quarto templates
├── data/experimental/        # Placeholder for experimental CV/CA data
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Output Directory Layout

After running a campaign, the output structure is:

```
data/
├── simulated_raw/<campaign_id>/
│   └── <run_id>/
│       ├── params.json       # Full configuration snapshot
│       ├── metadata.json     # Runtime metadata + status
│       ├── metrics.json      # Final AI, UI, growth rate, etc.
│       ├── fields.npz        # Concentration snapshots
│       ├── thickness.npz     # Film thickness snapshots
│       ├── timeseries.csv    # Per-timestep diagnostics
│       ├── figures/          # Per-run plots
│       └── logs/run.log      # Detailed run log
├── simulated_agg/<campaign_id>/
│   ├── results.csv           # Aggregated metrics (one row per run)
│   ├── descriptive_stats.csv
│   ├── AI_anova_table.csv
│   ├── UI_anova_table.csv
│   ├── pareto_front.csv
│   └── best_conditions.csv
└── reports/<campaign_id>/
    ├── final_report.html
    ├── final_report.pdf      # (if weasyprint/wkhtmltopdf available)
    └── figures/              # Campaign-level plots
```

## Factorial Design

The default factorial design crosses:

| Factor | Levels | Values |
|--------|--------|--------|
| Frequency | 3 | 1, 10, 100 Hz |
| Diffusion coefficient (D) | 3 | 1e-10 (low), 6.5e-10 (mid), 2e-9 (high) m^2/s |
| Geometry (bridge width) | 2 | narrow (2 mm), wide (5 mm) |
| Replicates | 3 | Seeds 42, 43, 44 |

**Total: 3 x 3 x 2 x 3 = 54 runs**

D levels correspond to electrolyte viscosity via the Stokes-Einstein relation: D = kT / (6 pi mu r).

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Research software. Please cite if used in publications.
