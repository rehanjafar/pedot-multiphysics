# EcoVolt / pedot-multiphysics

Research software for **directly driven gold wires growing PEDOT dendrites**, with a separate legacy two-compartment deposition benchmark. The models are exploratory and uncalibrated. They cannot yet establish physical manufacturing optima, material performance or energy savings.

Read [Scientific basis and readiness](docs/SCIENTIFIC_BASIS.md) for primary papers, verified experimental context, equations, parameter assumptions, units and the measurements needed for validation. The direct-wire model updates the conducting geometry and electric field as deposits grow. The legacy fixed-strip model does not.

## Install and test

Python 3.11+; from the repository directory:

```sh
python -m pip install -e '.[dev]'
python -m pytest tests --cov=pedot_sim --cov-report=term-missing
```

GitHub checks use Ubuntu 22.04 and Python 3.11–3.15. Release 0.4 requires Python 3.11+ because Python 3.9/3.10 are [past upstream support](https://devguide.python.org/versions/); the development dependencies require the patched pytest 9.0.3 or later. Tests cover analytic diffusion/convergence, potential boundary conditions, conservation, waveform timing, direct-wire geometry feedback, seeded growth, invalid inputs, measurement comparisons and output workflows. Passing these tests is not experimental validation.

See [Verification results](docs/VERIFICATION_RESULTS.md) for the checked benchmark, reproducible commands, test scope and remaining evidence gaps.

## Direct-wire dendrite research

```sh
pedot-sim run-dendrite --config configs/example_dendrites.json --run-id seed42 --campaign-id wires_v1 --output-base data
pedot-sim analyze-dendrites --campaign-id wires_v1 --output-base data
```

Outputs are under `data/dendrites/wires_v1/`: each run has parameters, metadata, morphology snapshots, a morphology figure, timeseries and metrics. Campaign analysis writes `results.csv`, descriptive statistics, censor-aware `bridging_summary.csv` and `report.html`. Change the configuration seed and use a new run ID for independent realizations. Reusing a seed/condition is rejected as replication during analysis.

The lattice model simulates mobile charged aggregates, drift/diffusion, irreversible attachment, equipotential deposits and field feedback. It stops at the first conducting bridge. The left wire is driven and the right is grounded. All geometric quantities are in **lattice cells** and elapsed time in **model steps**. `period_steps` is not a frequency in Hz, and particle count is not molar concentration. Mobility, attachment and activation settings are assumptions. No physical length/time calibration or explicit counter-reaction is included. The example uses effective anionic carriers, not negatively charged EDOT. `integration_dt` controls time refinement; the integrator also resolves waveform transitions and at least 40 intervals per period. See the scientific-basis document before comparing maps with laboratory images.

For independent period/seed ensembles:

```sh
pedot-sim run-dendrite-batch --config configs/example_dendrites.json --campaign-id wire_ensemble_v1 --periods 20 40 80 --replicates 5 --output-base data
```

The saved plan supports resuming complete runs with identical code/settings. The bridging summary reports model probability with a Wilson interval, censored counts and restricted mean time. It does not count an unbridged run as a successful completion.

## Fixed-strip numerical benchmark

```sh
pedot-sim run-single --config configs/example_verify.json --run-id one --campaign-id strip_v3 --output-base data
pedot-sim analyze --campaign-id strip_v3 --output-base data
pedot-sim make-report --campaign-id strip_v3 --output-base data
pedot-sim verify --config configs/example_verify.json --campaign-id verification_v4 --mesh-sizes 100 200 400 --output-base data
```

Stage A imposes `J=kC` independently of voltage. B adds phenomenological field enhancement; C adds species migration (neutral EDOT defaults to charge zero); D uses a floating signed Butler–Volmer proxy and anodic deposition. These stages describe the **legacy floating-strip setup**, not directly driven wires. They omit explicit cathodic chemistry and growing geometry. The supplied stiff Stage A verification case uses a boundary-fitted graded mesh (`mesh_stretch=10`) and passes the unchanged 5% tolerance at the documented levels. Inspect the saved verification summary; this numerical result does not establish validity for other stages or experiments.

Transport couples masked theta diffusion and implicit consumption in one solve, followed by upwind migration. Backward Euler (`implicit_theta=1`) is the default; theta=0.5 requests Crank–Nicolson. The coupled splitting is first order. Waveform-dependent runs use at least 40 steps per period and include pulse/rest transitions as endpoints. Midpoint drive sampling and a final partial step prevent phase/end-time mistakes. `adaptive_dt=true` is rejected; `cfl_safety` controls migration substeps.

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

Published reference data are bundled as described below; no matched calibration or independent validation dataset for these models is supplied. For thickness validation, supply simulation and experimental CSVs with `time,mean_h` in seconds/metres, plus metadata such as:

```json
{"source": "Your dataset identifier or DOI", "units": {"time": "s", "mean_h": "m"}}
```

```sh
pedot-sim validate --simulation data/simulated_raw/strip_v3/one/timeseries.csv --experiment measurements.csv --metadata measurements.json --observable thickness --max-rmse 1e-7 --output-dir data/comparisons/thickness
```

The shown threshold is illustrative, not a scientific acceptance criterion. Set it before comparison based on the experiment and its uncertainty. Profiles use `x,h` [m,m]; explicit current traces use `t,I` [s,A], and separately split monotonic CV branches use `V,I` [V,A]. Molar flux cannot be silently converted to amperes. The direct-wire model's lattice results need a justified physical mapping before quantitative measurement comparison.

Reports retain error metrics, source metadata and input hashes; nonfinite values and extrapolation are rejected. A comparison without a threshold is marked `not_assessed`. A dataset-specific pass is not universal model validation. Keep calibration and validation measurements separate.

## Analyze measured manufacturing data

```sh
pedot-sim analyze-manufacturing --trace measurements/electrical.csv --metadata measurements/process.json --output-dir data/measured_process
```

CSV columns are `time_s,voltage_V,current_A`. Metadata requires a source identifier and explicit units:

```json
{"source":"Your experiment ID", "units":{"time_s":"s","voltage_V":"V","current_A":"A"}}
```

Optional metadata fields are `dry_product_mass_kg`, `initial_monomer_mass_kg`, `film_area_m2`, `mean_thickness_m`, `four_terminal_resistance_ohm`, `voltage_probe_spacing_m`, `conducting_width_m` and `frequency_Hz`. Supply only independently measured quantities. Outputs include net/imported/returned terminal energy and charge; optional inputs enable specific energy, mass ratio, effective density and rectangular-film conductivity calculations. Irregular dendrites require their own electrical geometry model. The source files are hashed and assumptions accompany the HTML/CSV/JSON reports. No process measurement dataset is invented; the separately bundled published reference data do not establish your process energy or yield.

## Published evidence and visual outputs

```sh
pedot-sim analyze-literature --output-dir data/published_evidence
pedot-sim visualize-run --run-dir data/dendrites/wires_v1/seed42
```

Open `data/published_evidence/index.html` for seven purposeful figures from **2,138 published records** across six numerical tables: amplitude and frequency morphology, offset asymmetry, growth histories, impedance and first-cycle voltammetry. They are curated from [Janzakova et al. (2021)](https://doi.org/10.1038/s41467-021-27274-9) and the [versioned numerical dataset](https://doi.org/10.6084/m9.figshare.16814710.v1), with source hashes, row/column provenance, conversions and CC BY 4.0 attribution. These observations are reference evidence; they do not calibrate the independent model.

Every new simulation produces `views/index.html` with PNG and editable SVG figures, offline interactive 3D views, numerical arrays and captions explaining the purpose and limits. Direct-wire runs also produce an OBJ network mesh, the last solved potential **with its pre-attachment geometry and midpoint time**, and growth snapshots. Fixed-strip runs save every integration interval's material balance and an actual-time thickness history. Enabling `output.save_animations` generates a GIF from recorded profiles when at least two snapshots exist. `visualize-run` re-renders saved states without rerunning physics.

The 3D network is a one-cell illustrative extrusion, not a measured volume or a 3D transport solver. Thickness-history surfaces use time as an axis. Published growth traces use time, voltage and projected area; they do not interpolate a surface between measured conditions. Preserve captions and assumptions when exporting. Keep each gallery folder and its local `plotly.min.js` together for offline use.

Read [Data and visualization quality](docs/DATA_AND_VISUALIZATION.md) for source limitations, the flagged 10 Hz anomaly, correct interpretation of error bars, exact output locations and reproduction commands.

## Output dependencies and license

Static and interactive 3D output and GIF animation are core features; no PyVista/OpenGL installation is required. The empty `viz3d` extra is retained for installation compatibility. `io` supplies optional columnar/movie tools; PDF reports require an optional renderer. Dependency ranges are not a lockfile or a claim of zero vulnerabilities.

Software: [MIT](LICENSE). Bundled Janzakova numerical data: [CC BY 4.0 with attribution and modifications](src/pedot_sim/datasets/janzakova2021/ATTRIBUTION.md). Cite the software revision, configuration, primary sources, resolution and independent validation evidence in scientific use.
