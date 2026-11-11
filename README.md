# EcoVolt / pedot-multiphysics

Research software for **directly driven gold wires growing PEDOT dendrites**, with a separate legacy two-compartment deposition benchmark. The models are exploratory and uncalibrated. They cannot yet establish physical manufacturing optima, material performance or energy savings.

Read [Scientific basis and readiness](docs/SCIENTIFIC_BASIS.md) for primary papers, verified experimental context, equations, parameter assumptions, units and the measurements needed for validation. The direct-wire model updates the conducting geometry and electric field as deposits grow. The legacy fixed-strip model does not.

## Install and test

Python 3.9+; from the repository directory:

```sh
python -m pip install -e '.[dev]'
python -m pytest tests --cov=pedot_sim --cov-report=term-missing
```

GitHub checks use Ubuntu 22.04 and Python 3.9, 3.12 and 3.14. The OS is explicit because Python 3.9 is unavailable on the latest Ubuntu runner. Tests cover analytic diffusion/convergence, potential boundary conditions, conservation, waveform timing, direct-wire geometry feedback, seeded growth, invalid inputs, measurement comparisons and output workflows. Passing these tests is not experimental validation.

## Direct-wire dendrite research

```sh
pedot-sim run-dendrite --config configs/example_dendrites.json --run-id seed42 --campaign-id wires_v1 --output-base data
pedot-sim analyze-dendrites --campaign-id wires_v1 --output-base data
```

Outputs are under `data/dendrites/wires_v1/`: each run has parameters, metadata, morphology snapshots, a morphology figure, timeseries and metrics. Campaign analysis writes `results.csv`, descriptive statistics and `report.html`. Change the configuration seed and use a new run ID for independent realizations. Reusing a seed/condition is rejected as replication during analysis.

The lattice model simulates mobile charged aggregates, drift/diffusion, irreversible attachment, equipotential deposits and field feedback. It stops at the first conducting bridge. The left wire is driven and the right is grounded. All geometric quantities are in **lattice cells** and elapsed time in **model steps**. `period_steps` is not a frequency in Hz, and particle count is not molar concentration. Mobility, attachment and activation settings are assumptions. No physical length/time calibration or explicit counter-reaction is included. See the scientific-basis document before comparing maps with laboratory images.

## Fixed-strip numerical benchmark

```sh
pedot-sim run-single --config configs/example_verify.json --run-id one --campaign-id strip_v3 --output-base data
pedot-sim analyze --campaign-id strip_v3 --output-base data
pedot-sim make-report --campaign-id strip_v3 --output-base data
pedot-sim verify --config configs/example_verify.json --campaign-id verification_v3 --output-base data
```

Stage A imposes `J=kC` independently of voltage. B adds phenomenological field enhancement; C adds species migration (neutral EDOT defaults to charge zero); D uses a floating signed Butler–Volmer proxy and anodic deposition. These stages describe the **legacy floating-strip setup**, not directly driven wires. They omit explicit cathodic chemistry and growing geometry. The default strip verification case is stiff and has previously failed mesh tolerance: inspect the saved verification summary rather than treating successful execution as scientific accuracy.

Transport uses masked theta diffusion, upwind migration and local implicit consumption. Backward Euler (`implicit_theta=1`) is the default; theta=0.5 requests Crank–Nicolson. The coupled splitting is first order. Waveform-dependent runs use at least 40 steps per period and include pulse/rest transitions as endpoints. Midpoint drive sampling and a final partial step prevent phase/end-time mistakes. `adaptive_dt=true` is rejected; `cfl_safety` controls migration substeps.

The dry neutral repeat mass defaults to 0.14016 kg/mol; film density 1340 kg/m³ and deposition efficiency 1 are configurable **assumptions**. Every run checks consumed monomer, net reservoir exchange and retained film mass. Outputs include `research_audit.json`, an always-saved `final_state.npz`, thickness CV, signed left/right imbalance, dimensioned mass/charge proxies and electrolyte-only ohmic energy. Extensive quantities are **per metre of out-of-plane depth**. They are not measured cell current, total process energy or experimentally established yield. `save_fields` controls snapshot archives; final state and diagnostics remain available.

AI retains the legacy peak/mean thickness definition; UI is standard deviation in metres. Neither measures dendrite branching. Zero-growth states are excluded from coating rankings. Default stochastic heterogeneity is disabled; when enabled it changes requested flux while preserving material accounting. `noise_amplitude` remains a legacy unused field, not a source of free film mass.

## Campaigns and statistics

```sh
pedot-sim run-batch --config configs/example_factorial.json --campaign-id factorial_v3 --sweep factorial --max-workers 4
pedot-sim run-batch --config configs/example_dense.json --campaign-id dense_v3 --sweep dense --max-workers 4
pedot-sim analyze --campaign-id dense_v3 --output-base data
pedot-sim make-report --campaign-id dense_v3 --output-base data
```

Factorials honor the requested frequency, D and geometry levels. Dense Latin hypercube/random designs sample five dimensions: frequency, log(D), width, DC offset and AC amplitude. Grid sampling uses a three-dimensional frequency/D/width grid with fixed configured voltages; its actual size is the nearest cube with at least two levels per axis. Numeric bridge width is authoritative. Diffusion/viscosity and conductivity/ionic strength are independent parameters, not inferred solvent relationships.

Descriptive summaries use exact numeric conditions. ANOVA is skipped with a recorded reason for deterministic, incomplete or unsuitable designs. Standardized linear surrogates report leave-one-condition-out error and do not imply causality or empirical validation. Candidate/Pareto rankings use explicit arbitrary objectives and do not prove manufacturing optimality.

A changed source/configuration fingerprint requires a new campaign ID. Existing run directories are never overwritten, including partial failures. Manifests are written atomically; IDs and escaping symlinks are rejected. A failed batch or numerical verification returns a nonzero exit code. JSON undefined numeric quantities become `null`.

## Compare measurements

No experimental datasets are bundled. For thickness, supply simulation and experimental CSVs with `time,mean_h` in seconds/metres, plus metadata such as:

```json
{"source": "Your dataset identifier or DOI", "units": {"time": "s", "mean_h": "m"}}
```

```sh
pedot-sim validate --simulation data/simulated_raw/strip_v3/one/timeseries.csv --experiment measurements.csv --metadata measurements.json --observable thickness --max-rmse 1e-7 --output-dir data/comparisons/thickness
```

The shown threshold is illustrative, not a scientific acceptance criterion. Set it before comparison based on the experiment and its uncertainty. Profiles use `x,h` [m,m]; explicit current traces use `t,I` [s,A], and separately split monotonic CV branches use `V,I` [V,A]. Molar flux cannot be silently converted to amperes. The direct-wire model's lattice results need a justified physical mapping before quantitative measurement comparison.

Reports retain error metrics, source metadata and input hashes; nonfinite values and extrapolation are rejected. A comparison without a threshold is marked `not_assessed`. A dataset-specific pass is not universal model validation. Keep calibration and validation measurements separate.

## Optional output and license

`pip install -e '.[viz3d,io]'` enables optional animation dependencies. Legacy 3D views extrude a one-dimensional thickness profile; they are not three-dimensional physics. PDF reports require an optional renderer. Dependency ranges are not a lockfile or a claim of zero vulnerabilities.

MIT; see [LICENSE](LICENSE). Cite the software revision, complete configuration, primary scientific sources, numerical resolution and independent validation evidence in scientific use.
