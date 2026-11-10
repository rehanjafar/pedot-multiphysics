# pedot-multiphysics

Research prototype for fixed-domain, waveform-driven PEDOT deposition surrogates.
**Uncalibrated: do not interpret results as experimentally validated morphology or cell current.**
The numerical and campaign fixes change results substantially; use new campaign IDs and regenerate older results.

## Install and run

Python 3.9+; install from this directory:

```sh
pip install -e '.[dev]'
pytest tests/
pedot-sim run-single --config configs/example_verify.json --run-id smoke_001 --campaign-id quick_test -v
pedot-sim run-batch --config configs/example_factorial.json --campaign-id factorial_v2 --max-workers 4 --sweep factorial -v
pedot-sim run-batch --config configs/example_dense.json --campaign-id dense_v2 --sweep dense --max-workers 4
pedot-sim verify --config configs/example_verify.json --campaign-id verify_v2
pedot-sim analyze --campaign-id factorial_v2 --output-base data
pedot-sim make-report --campaign-id factorial_v2 --output-base data
```

`verify` runs deterministic mesh/time studies and exits unsuccessfully when the finest two levels differ by more than 5% (with small absolute floors), fail, or do not represent distinct timesteps. Passing these output tolerances is **not** experimental validation or a universal error bound. Richardson estimates use observed order and actual spacings, and are absent when not identifiable. Only the supplied physics stage is verified; test each stage and waveform independently. Repeated runs need new IDs: existing output is never silently overwritten.

A deterministic factorial has no independent replicate noise: ANOVA significance is not meaningful in that case. Continuous dense sweeps are not a factorial ANOVA design. Analysis rejects insufficient/rank-deficient inferential designs; descriptive simulation comparisons still require physical validation. Composite/Pareto scores describe the selected objectives and their weights, not proven experimental optimality.

## Model and units

- Stage A imposes a first-order surface-reaction surrogate, J=k C, independent of voltage.
- Stage B adds a prescribed Laplace field and phenomenological field enhancement J=k C (1+beta |E|). It remains a surrogate, including at zero voltage.
- Stage C additionally transports a species with the configured charge number. The default is neutral EDOT (z=0); changing it requires justification of the chemical species and reactions.
- Stage D uses phenomenological Butler–Volmer kinetics. A scalar metal potential balances the **signed proxy current** with eta=metal−solution and zero equilibrium reference. Only anodic current consumes monomer/deposits polymer. The cathodic branch is a mathematical balancing proxy, **not a calibrated counter-reaction**. The prescribed electrolyte field does not respond to interfacial current. This weak-coupling approximation cannot establish self-consistent BPE polarization or measured cell current.

Diffusion uses a sparse masked theta scheme: theta=0 explicit, 0.5 Crank–Nicolson, 1 backward Euler. Inactive faces are insulated; feeder cells are fixed-concentration reservoirs. Reaction is split with local backward Euler, so the coupled method is first order in time. Migration uses conservative upwind face fluxes with positivity substeps. Unstable explicit steps, nonpositive diffusion solutions and unconverged Laplace fields fail instead of being silently clipped/accepted. Reduce dt or use theta=1 if Crank–Nicolson develops oscillations.

Each electrode row cell represents **one effective exposed interface per unit out-of-plane depth**, with surface/volume ratio 1/dy. Film mass is derived from the actual consumed monomer, using the approximate EDOT molar mass and PEDOT density. This is not a resolved two-sided solid electrode. Film thickness does not alter geometry, electrical fields or transport.

For field-dependent stages the runner automatically caps dt at 1/(40 f), and more tightly for short pulse duty intervals; dt in the config is an upper limit. The final partial step ends at t_end; snapshots include the initial state. Inspect effective_dt in outputs and establish waveform convergence at finer timesteps. These caps are not a proof of accuracy. Legacy adaptive_dt/cfl_safety fields are retained for input compatibility, not a separate adaptive error controller.

Default stochastic noise is disabled. When enabled, spatial Poisson events modulate requested reaction flux; film still consumes monomer. The old additive thickness-noise API is rejected. This is an uncalibrated heterogeneity model, not resolved nucleation/coalescence. noise_amplitude is a legacy configuration field and no longer creates initial polymer mass. Seed reproducibility does not replace stochastic ensemble convergence.

All physical values are SI. Surface flux is mol/(m² s); integrated flux is mol/(m s), per unit out-of-plane depth. It is not amperes. AI=max(h)/mean(h) includes bare electrode positions; UI=std(h) is in meters. Zero-growth states must not be interpreted as a useful uniform coating. timescale_ratio=L_electrode k/D; diffusion_pulse_ratio=(bridge_width/2)² f/D is a separate quantity. Neither proves a frequency/uniformity hypothesis.

## Parameters and validation limits

Material constants and kinetic defaults are exploratory values, not traceably fitted PEDOT parameters. k, j0, beta, density, species identity, equilibrium potentials, deposition efficiency and counter-reaction chemistry require measurement and parameter provenance before quantitative use. D and viscosity are independent inputs; the Stokes–Einstein helper provides an explicitly idealized spherical-solute estimate, not an empirical calibration. Configured conductivity/ionic_strength do not change the prescribed-voltage, uniform-conductivity Laplace solution. No convection, multistep oligomer chemistry, double-layer charging, nodule geometry or field feedback from growing film is modeled.

No experimental data are bundled. The validation Python interface requires dimensioned simulation columns t [s] and I [A] (or V and I for separately aligned monotonic CV branches). Native output supplies time and molar flux: it intentionally cannot be compared directly to amperes. Supply an independently justified current/geometry conversion or a calibrated observable first. Failed validation returns an explicit failed status; interpolation outside the simulated range or across nonmonotonic CV branches is rejected.

## Campaigns, output and safety

The factorial honors configured frequencies, D_levels, geometries and replicates. The default design is 3×3×2×3=54 runs. Dense sampling honors sample count and strategy, samples frequency/D/width, and independently samples configured DC/amplitude ranges. Numeric bridge_width is authoritative; geometry_type is a label and never overrides it. Explicitly specify the width for a single custom geometry.

Outputs are under data/simulated_raw/<campaign>/<run>, data/simulated_agg/<campaign>, data/verification/<campaign> and data/reports/<campaign>. Campaign manifests and source/configuration fingerprints prevent changed campaigns from silently resuming old results. Use a new campaign ID after changes. Completed entries require saved metrics and completed metadata; failures produce a nonzero batch exit. Failed partial runs require a new campaign ID rather than destructive automatic retries. IDs are restricted to safe names; existing runs are refused.

save_fields controls concentration archives. Thickness/timeseries/metrics remain available. Animations require snapshots and optional visualization/I/O dependencies (`pip install -e '.[viz3d,io]'`). The 3D view merely extrudes the 1D profile and is not a 3D physical simulation. Reports include actual configurations, escaped labels and packaged templates; PDF rendering is optional and not part of the scientific verification tests. Dependency ranges are supported ranges, not a lockfile or vulnerability audit.

## License

MIT; see LICENSE. Scientific use should identify the software revision, full configuration, environment, model limitations and independent validation evidence.
