# Scientific correctness and readiness fixes

These changes invalidate quantitative comparisons to output from the original 0.1.0 source. Start new campaigns.

Implemented:

- Conservative insulated diffusion, a genuine sparse theta solve, positivity checks, conservative upwind migration with subcycling, and masked field gradients.
- Surface-flux/cell-volume conversion using one effective exposed interface per electrode cell; local implicit reaction and film growth use identical consumed monomer.
- Anodic-only deposition; signed, concentration-weighted floating-potential proxy balance. This remains weak coupling, not a self-consistent electrode/electrolyte solution.
- Neutral monomer default; stochastic deposition disabled by default. Optional Poisson events modulate requested flux and never directly create film mass.
- Automatic waveform resolution limits, correct final partial steps and snapshot timestamps, growth metrics independent of saved snapshots, and separate reaction/diffusion versus pulse/diffusion ratios.
- Full-profile anisotropy metric, zero-growth exclusion from optimization, finite-metric checks and inferential design/rank checks.
- Authoritative numeric geometry, fractional-frequency IDs, configured sweep dimensions/ranges/counts, source/configuration campaign fingerprints, output existence/status checks, immutable existing runs and failed-batch exit status.
- Consistent manifest paths, safe IDs, escaped reports, bundled templates, actual run configurations in methods, removed external report scripts, field-output flag support and animation wiring.
- Deterministic verification, actual spacings and observed-order estimates, explicit convergence tolerances and failure status. Validation requires dimensioned current and rejects unsupported interpolation.
- Corrected documentation, clear model/parameter limitations, MIT text consistent with existing metadata, ignored generated files, and a Python 3.9/3.12 CI test workflow.

Validation on macOS ARM64/Python 3.14:

- 68 tests passed, including 24 regression/invariant cases added to the original 44 tests. Eight existing concentration-plot layout warnings remain.
- Source and wheel builds succeeded; the wheel includes its report template.
- Supplied single-run CLI accepted trailing -v and completed; a real two-worker campaign honored its configured single D/geometry combination; aggregate/HTML report generation succeeded.
- The supplied deterministic verification run now **fails** its mesh tolerance, rather than declaring success: AI changes from 1.04239 at 100x100 to 1.12178 at 200x200. Its timestep sequence is much more stable. This requires further mesh studies; the tolerance was not relaxed to force a pass.

Still outside the claims of this prototype:

- A calibrated cathodic counter-reaction and self-consistent electrolyte/BPE interfacial-current coupling.
- Traceably measured material/kinetic parameters, deposition efficiencies, real experimental validation and a justified current observable for CA/CV comparison.
- Resolved evolving electrode geometry, nodules, oligomer chemistry, convection or double-layer charging.
- Calibration and ensemble/spatial convergence of the optional stochastic heterogeneity model.
- Production-size campaigns, optional PyVista/ffmpeg/PDF validation, dependency CVE auditing and cross-version CI execution (workflow added, not executed locally).

These limitations are explicit in the README and generated reports. Passing the test suite does not make the scientific model experimentally validated.

## 0.2.0 — direct-wire research model and expanded verification

Research target clarified as directly driven gold wires, rather than floating
bipolar electrodes. Added an independent charged-aggregate lattice model with
conducting growth, repeated Laplace solutions, seeded drift/diffusion, aggregate
conservation, attachment activation and termination at electrical contact.
Added direct-wire campaign reports. Lattice cells/model steps have no calibrated
physical mapping; no empirical accuracy or universal manufacturing optimum is claimed.

Corrected overpotential sign and neutral repeat mass; exposed assumed film
density and retention efficiency. Added full material accounts, fixed-strip
electrolyte heating diagnostics, resolution warnings and research audits.
Waveform steps include pulse/rest transitions. Dense Latin hypercubes now cover
all five sampled factors; statistical analyses distinguish valid stochastic
factorials, continuous designs and deterministic repeats. Added source/units/hash
provenance for experimental comparisons and runtime source/dependency fingerprints.

JSON/manifest writes are atomic, runs claim their directories exclusively and
escaping symlinks are rejected. CI pins reviewed action revisions and Ubuntu
22.04, with Python 3.9/3.12/3.14 tested independently. Added primary-source
scientific documentation and explicit unsupported manufacturing properties.

Validation at implementation: 174 tests passed locally (previous PR: 68), with
68% overall statement coverage and 90%+ in the principal growth/transport/field
modules and direct-wire model. Source/wheel builds and direct-wire CLI/report
workflow succeeded. Numerical refinement and empirical validation remain separate
requirements; consult generated verification summaries and research audits.
