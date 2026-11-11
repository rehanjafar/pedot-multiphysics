# Verification record — 0.4.0

Checked 11 November 2026. This record distinguishes tested numerical behavior from experimentally validated predictions. No fitted experimental dataset is claimed.

## Software and package checks

- 264 automated tests pass locally on Python 3.14.7; statement coverage is 77%. Numerical transport, growth, measurement accounting and direct-wire core paths have substantially higher coverage. New scientific galleries, 3D geometry, actual exports and bundled data are tested. Some legacy plotting helpers remain untested.
- Analytic tests include a simultaneous two-cell reaction/diffusion solution, masked diffusion eigenmodes, rectangular Laplace/Joule-loss solutions, conservation on graded grids, area-weighted floating-current balance, exact integrals of piecewise-linear electrical signals and drive transition timing.
- Workflow tests exercise independent seeds, conservation/bridging, censor-aware ensemble summaries, resume protection, HTML escaping, units/provenance, invalid inputs, CLI workflows and all four strip stages on graded grids.
- The full 264-test suite also passes with pandas 3.1.1; the pandas upper bound now admits the supported 3.x series, avoiding a source build of an old release on Python 3.15.
- Ruff's F checks pass. Source and wheel distributions build; installed dependencies pass `pip check`.
- `pip-audit` reports no known vulnerabilities in the checked local environment after updating pytest to a patched release. The former pytest requirement allowed CVE-2025-71176; the development requirement is now pytest >=9.0.3. See the [reviewed advisory](https://github.com/advisories/GHSA-6w46-j5rx-g56g). This is a time-specific package audit, not proof of zero vulnerabilities or a scan of every optional environment.
- CI tests supported Python 3.11–3.15 and runs the complete strip refinement study separately. Each PR's GitHub checks are the authoritative status of that revision.

## Stiff strip refinement

Command from the repository root:

```sh
pedot-sim verify --config configs/example_verify.json --campaign-id verification_v4 --mesh-sizes 100 200 400 --output-base data
```

The physical assumptions are unchanged: Stage A, D=6.5e-10 m²/s, k=1e-3 m/s, 50 mol/m³ bulk concentration and 10 s duration. Diffusion and linear reaction are solved simultaneously; theta=1. The mesh uses boundary-fitted sinh grading with strength 10. Time refinement uses the finest spatial level. The output acceptance tolerance remains 5%.

| Spatial budget | Mean thickness (µm) | Peak/mean ratio | Thickness standard deviation (µm) |
|---:|---:|---:|---:|
| 100 × 100 | 0.921459 | 7.010197 | 0.167120 |
| 200 × 200 | 0.949316 | 7.186312 | 0.174976 |
| 400 × 400 | 0.956535 | 7.301616 | 0.177039 |

The finest two grids differ by approximately 0.755%, 1.579% and 1.165%, respectively. Time steps 0.01, 0.005 and 0.0025 s give finest-pair differences below 0.004% for all three quantities. Material-balance residuals are below 2e-15 mol per metre of depth in these runs. [Raw mesh results](verification/mesh_refinement.csv), [time results](verification/timestep_refinement.csv) and the [acceptance summary](verification/verification_summary.json) are retained.

These results establish agreement for this benchmark and these output measures. They do not establish accuracy for every physics stage, arbitrary geometry or quantitative PEDOT manufacture. The fixed electrode remains an infinitesimal reaction-strip approximation; strong local growth and missing interfacial/counter-reaction physics still limit physical interpretation.

## Direct-wire execution evidence

Thirty independent runs of `configs/example_dendrites.json`, ten seeds at each period of 20, 40 and 80 model steps, completed through the real batch workflow. The [ensemble summary](verification/dendrite_ensemble_summary.csv) retains bridge probability with Wilson intervals, censor counts and restricted mean time. These are model outcomes in lattice/model units, not measured fabrication performance or a validated frequency optimum. Full figures, metadata and morphology arrays can be regenerated with:

```sh
pedot-sim run-dendrite-batch --config configs/example_dendrites.json --campaign-id wire_ensemble_v1 --periods 20 40 80 --replicates 10 --output-base data
```

## Evidence still required

Quantitative research conclusions require matched measurements from the intended directly driven gold-wire system: wire/gap dimensions, bath chemistry, calibrated voltage/current traces, image sequences with physical scale, deposited mass/composition and independent validation runs. The primary literature supplies numerical reference observations and mechanisms; its parameters do not calibrate this independent implementation. Conductivity, adhesion, porosity, ageing and lifecycle performance cannot be established by the existing growth model.


## Published data and depictions

The original 2,608,639-byte dataset was checked against its published MD5 and the recorded SHA-256 before selecting 2,138 records. The six packaged tables have individual hashes. Source values, units, first-cycle acquisition order, six-position sample SD and the unresolved 10 Hz zero area have regression checks. Reference observations remain distinct from source circuit simulations and fitted parameters.

Twenty-one additional checks cover oriented mesh volume/shared faces/OBJ indexing, missing and invalid inputs, actual PNG/SVG/HTML exports and linked files, offline JavaScript, plotted values versus CSV records, impedance coordinate identities, exact cumulative energy, saved pre-attachment fields, graded growth-rate weights, every-step mass budgets, and timestamped GIFs. The final 21-test subset was rerun after caption and depth-label improvements. Published 3D interaction was also inspected in a browser; static frequency, growth, network and boundary-layer figures were inspected visually.

All 30 independent direct-wire runs reached 400 model steps without a bridge. This is a censored observation: each period has 0/10 bridges and a Wilson 95% upper bound of about 0.278, not evidence that bridging is impossible. The six-run strip refinement study was repeated with all new required galleries and passed the unchanged 5% acceptance threshold. No physical 3D volume, experimental repeatability interval or calibrated manufacturing optimum is claimed.

The evidence package includes exact simulation-source and environment fingerprints, complete state arrays, every-step CSV budgets, 30 OBJ meshes, static/interactive galleries and a recorded-profile animation. Numerical archives retain their original provenance. Separately recorded rendering provenance identifies the final cosmetic depth-label revision; the source snapshot used for the simulations is retained and hash verified. The final published-data caption specifies that post-growth impedance was recorded in 1 mM NaPSS without EDOT/BQ.
