# Scientific basis and readiness

Reviewed 11 November 2026. Target research system: **directly driven gold wires growing PEDOT dendrites**, as clarified by the researcher. The original two-compartment floating-strip model is retained as a distinct numerical benchmark. Neither implementation is experimentally calibrated. Passing tests establishes software/numerical properties; it does not establish predictive manufacturing accuracy.

## Primary evidence and applicability

| Source | Verified observation or conditions | Consequence for this repository |
|---|---|---|
| [Kumar et al., Scientific Reports 12, 6395 (2022)](https://doi.org/10.1038/s41598-022-10082-6) | Directly driven wire/ground configuration; mesoscale charged-particle motion, attachment and recalculated potentials on growing conductors. The paper distinguishes qualitative model frequency from laboratory frequency. | Motivates the separate `dendrites.py` aggregate model. Our implementation is independent, uses exclusion-lattice hops, and is not a reproduction of their simulation or fitted data. |
| [Janzakova et al., Dendritic Organic Electrochemical Transistors (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8693061/) | Aqueous EDOT 10 mM, BQ 10 mM, NaPSS 1 mM; 25 µm Au wires separated by 240 µm in a 20 µL drop; 10 V peak-to-peak bipolar drive. Electrical characterization was measured separately. | These are experimental context, **not** parameter validation for the code. No film conductivity is inferred from thickness or branch shape. |
| [Koizumi et al., Nature Communications 7, 10404 (2016)](https://doi.org/10.1038/ncomms10404) | Wireless Au BPEs between Pt feeders; MeCN with 50 mM EDOT, 5 mM BQ and 1 mM supporting electrolyte. Oxidation and sacrificial reduction occur on opposite regions. | This is a different electrical boundary condition from directly driven wires. The floating-strip BV proxy lacks explicit BQ chemistry and is not a faithful reproduction. |
| [Lee et al., Advanced Functional Materials 34, 2406229 (2024)](https://doi.org/10.1002/adfm.202406229) | Dual floating patterned BPEs; MeCN with 150 mM EDOT, 15 mM BQ and 3 mM electrolyte. Seeding uses 50 V RMS AC; propagation about 60 V RMS plus 1–1.5 V DC, at 3 Hz. | Do not transplant these amplitudes or an optimum into a driven-wire aqueous cell. The legacy `asymmetric_adc` half-sine is an invented waveform family, not the paper's protocol. |
| [Janzakova et al., Nature Communications 12, 6898 (2021)](https://doi.org/10.1038/s41467-021-27274-9), [source dataset v1](https://doi.org/10.6084/m9.figshare.16814710.v1) | Numerical voltage/frequency/offset morphology, image-derived growth traces, impedance and voltammetry. Branching error bars describe six image positions, not six independently grown samples. | Six experimental/image-analysis subsets are bundled with original row/column provenance and CC BY 4.0 attribution. The 10 Hz zero-area anomaly is preserved and flagged. Circuit simulations and fitted circuit parameters are excluded from measurement tables. |
| [Baron et al., Chemical control for morphogenesis (online 2025; volume 4, 2026)](https://doi.org/10.1039/d5lp00225g) | Electrolyte, redox resources and monomer composition affect morphology and electrical response. The polymer's finite electrical resistance contributes to voltage drops, particularly at low salt concentration. | Ideal equipotential deposits cannot capture those chemistry-dependent resistance effects. Geometry alone does not determine conductivity; no universal frequency optimum or transferable recipe is justified. |
| [NIST fundamental constants](https://physics.nist.gov/cuu/Constants/index.html) | SI defining values for elementary charge, Avogadro and Boltzmann constants. | `F=N_A e` and `R=N_A k_B`; constants are stored at double precision. |
| [PubChem EDOT](https://pubchem.ncbi.nlm.nih.gov/compound/4421864) | EDOT molecular mass 142.18 g/mol. | Monomer mass is distinct from the approximate neutral polymer repeat C6H4O2S, 140.16 g/mol. The repeat value is a stoichiometric calculation after loss of two H, not a measured doped-film composition. |

The [EcoVolt project profile](https://thepointer.com/article/2026-05-03/brampton-teen-s-curiosity-leads-to-greener-electronics-breakthrough-targets-harm-of-everyday-devices) supplied project context only. It is not scientific validation or evidence that this repository implements every reported capability. Version 0.4 adds selected published numerical reference observations from the Janzakova dataset below. No measured 3D morphology, fitted kinetic calibration or held-out dataset for this independent implementation is supplied.

## Direct-wire model

`run-dendrite` uses two ideal conducting wires in an insulating rectangular bath. The left wire follows a square waveform and the right wire is grounded. Deposits inherit the electrical ownership of the wire they touch. A unit-voltage Laplace solution is recomputed whenever deposits change the geometry, then scaled by the current drive. There are no remote feeder electrodes in this model.

Each mobile aggregate occupies one vacant lattice cell. Hopping rates are symmetric diffusion plus upwind drift down the potential gradient for positive charge. Substeps bound the sum of move probabilities by 0.8. Exclusion, reflecting walls and rejected solid-cell hops preserve the number of aggregates. Each integration interval, an aggregate adjacent to an oxidizing interface may attach, subject to the configured positive metal-minus-solution activation voltage. Attachment decisions use the field and geometry before that step's update. The reference attachment probability is per unit model time; an interval of length dt uses `1-(1-p)^dt`. Drive integration uses the smaller of `integration_dt` and `period_steps/40`, inserts every polarity transition, and samples the midpoint. Snapshot archives retain their actual model timestamps. At first network contact, growth terminates: the code does not solve an ideal short between unequal imposed voltages.

The example selects effective negative carriers, which drift toward the anode as in the qualitative mechanism discussed by Kumar et al. The species identity is not established; sign remains configurable. This does not assign negative charge to neutral EDOT. Four-neighbor lattice transport/attachment has lattice-direction bias; the cited paper also discusses diagonal motion, double-layer screening and other assumptions absent here.

This is a phenomenological **aggregate** model. It does not assign the charge of an oligomer to neutral EDOT. Its diffusion/mobility, particle count, activation voltage and attachment probability require calibration. Voltage is in volts; geometry is in lattice cells and time in model steps. `period_steps` is not Hz. The number of model aggregates is not a concentration in mol/m³. A deposited cell is not a known polymer mass, dry thickness or pore size. No physical manufacturing energy can be derived from this model yet.

Reported quantities are occupied projected cells, projected coverage, deposited area on each wire, signed area imbalance, the bridging flag and elapsed model steps. An unbridged run's completion time is censored, not a successful completion time. Ensembles must vary seeds; repeating a seed is not replication. Do not infer a universal frequency/morphology trend from one stochastic realization. Test grid/particle-density/time-rate sensitivity after choosing a physical mapping, and compare **independent** measured images using the same segmentation and scale.

## Fixed-strip benchmark equations and units

The legacy `run-single` stages use an electrolyte mask, fixed reservoir concentration at feeder cells and insulated inactive faces. A graph Laplacian on the sampled grid approximates diffusion. With `mesh_stretch=0`, all active nodes carry equal numerical volume `dx*dy` per unit depth, including endpoints, and discrete measures differ from nominal dimensions. The recommended verification example instead uses a cell-centred mesh with exact edges at compartment, bridge and electrode boundaries. Symmetric sinh grading clusters cells at those edges; the resolution budget is distributed equally among anchored intervals. Every flux, inventory, electrode mean and Joule-loss sum uses actual face/volume weights. The field uses a sparse direct solve on graded grids (legacy uniform grids retain SOR). Reports expose actual cell counts and minimum/maximum widths. Grading does not remove the need for convergence studies.

- A: imposed surface flux `J=k C`; it grows even at zero voltage and is a transport/reaction benchmark.
- B: `J=k C (1+beta |E|)`, an uncalibrated field enhancement.
- C: B plus Nernst–Planck drift. Neutral EDOT has `z=0`, so C matches B by design.
- D: signed phenomenological BV, with `eta=metal-solution-Eeq`, zero-reference equilibrium and a floating metal voltage selected to balance signed **requested** proxy currents. Only positive current supplies deposition. The cathodic branch is not an explicit BQ reaction. The potential balance uses surface-area weights and previous-step concentration; subsequent implicit depletion need not preserve that electrical balance; no measured current claim is made.

The [electrochemical overpotential convention](https://doc.comsol.com/6.4/doc/com.comsol.help.fce/fce_ug_modeling.05.08.html) fixes the sign. BV exponential arguments saturate at ±50 for numerical protection; this is another reason high-voltage output is not quantitative kinetics.

Diffusion uses a theta scheme **coupled to** backward-Euler linear consumption in one matrix solve. Upwind migration follows as a separate step. Decoupling a stiff surface sink from diffusion caused a cell-size-dependent deposition error and has been removed. Backward Euler is the positivity-preserving default; Crank–Nicolson can produce oscillations at large steps and then fails explicitly. Implicit reaction, frozen concentration-dependent kinetic coefficients and migration splitting give first-order time accuracy even with theta=0.5. Drive values are sampled at interval midpoints; pulse/rest discontinuities are explicit step boundaries. Resolving the drive does not resolve electrochemical relaxation.

For one exposed face per electrode row cell, requested surface flux `J` [mol/(m² s)] implies a concentration sink `J/cell_height` [mol/(m³ s)]. Growth uses **actual** consumed material:

`deposited mass per depth = consumed mol per depth * retention efficiency * deposited molar mass`.

`film thickness increment = retained mol per area * deposited molar mass / effective film density`.

Density 1340 kg/m³, efficiency 1, diffusion 6.5e-10 m²/s, k=1e-3 m/s, j0=10 A/m², alpha=0.5 and beta=0.01 m/V are **assumptions**, not fitted values backed by these papers. Two electrons per repeat is an idealized conversion, excluding dopant charge and side reactions. Effective density/molar mass must describe the same measured film basis: dry neutral PEDOT, doped PEDOT:PSS and wet porous deposits are not interchangeable. Viscosity and ionic strength are metadata; D and conductivity are independent inputs. Stokes–Einstein is only an idealized helper with an assumed hydrodynamic radius.

The complete numerical inventory is checked each run:

`initial monomer + net reservoir input = remaining monomer + consumed monomer`.

`retained repeat amount = efficiency * consumed amount`.

Diffusive reservoir input is independently integrated across feeder/free-cell faces using the same theta time weighting, plus explicit imposed-reservoir inventory resets. This is compared with the total inventory change and reaction uptake. Migration reservoir resets are accounted separately. Analytic two-cell and closed-domain tests also check simultaneous reaction/diffusion and conservation. All extensive outputs are **per metre of out-of-plane depth**. Equivalent charge `n F consumed` is not terminal cell current. Ohmic heating sums `sigma*(potential jump)^2*face area/distance` on active edges and integrates in time. It excludes electrode polarization, double layers, chemistry, temperature feedback, supply losses and auxiliary process energy. Its ratio to deposited mass is a restricted model diagnostic, not a lifecycle metric or experimentally demonstrated energy saving.

AI is the legacy name for `max(h)/mean(h)`: a peak-to-mean thickness ratio, not a directional dendrite anisotropy measurement. UI is thickness standard deviation in metres. The dimensionless thickness CV and signed left/right mass imbalance are separate. Empty film has no useful coating uniformity; undefined ratios serialize as JSON null. Depletion scans stop at electrolyte walls. `sqrt(D/f)/dy < 3` produces a resolution warning, not an error estimate.

## Manufacturing coverage and missing evidence

| Research question | Current output | Needed for a defensible physical conclusion |
|---|---|---|
| Direct-wire morphology and bridging | Aggregate maps, projected area/imbalance, model-step completion | Measured wire shape/gap, image scale, replicate image sequences and calibrated particle/kinetic mapping |
| Thickness and deposition rate | Fixed-strip mass-consistent proxy | Thickness profiles, deposited mass, dopant/water fraction, effective density and independent kinetics |
| Transport limitation | Concentration, depletion and mesh warnings | Species identities, diffusion/mobility data, concentration and convection conditions |
| Electrical current and polarization | Prescribed potential; equivalent charge only | Explicit anodic/cathodic chemistry, impedance/double-layer model, reference potentials and current traces |
| Energy, yield and sustainability | Measured terminal energy/charge and product-mass normalization; separate fixed-strip proxy | Measured terminal V(t)I(t), product yield, solvent/supporting-salt use, washing/drying energy and lifecycle boundary |
| Conductivity/thermoelectric performance | Optional rectangular-film conductivity reduction from supplied four-terminal measurements; no prediction | Four-point conductivity, composition/doping, Seebeck and thermal measurements |
| Adhesion, porosity, mechanics, optical transparency, ageing, scale-up | No prediction | Appropriate material tests and additional validated physics |

## Validation workflow

1. Preserve raw measurements with source, instrument/protocol, uncertainty, units and calibration/validation split. Convert units explicitly before import. Never silently fit and validate on the same data.
2. Establish spatial, temporal, domain-size and stochastic ensemble convergence for each intended regime. Use the boundary-fitted example and explicit resolution levels (`--mesh-sizes 100 200 400`) for the stiff strip benchmark. Temporal refinement runs at the finest requested spatial level. The 5% output tolerance is unchanged. No Richardson estimate is claimed for the nonuniform family; saved output differences determine acceptance. The supplied Stage A benchmark passes, but different stages/parameters need their own studies.
3. Calibrate only identifiable parameters using designated training measurements. Include parameter uncertainty and plausible alternatives. This release does not supply an automatic parameter fit.
4. Compare held-out observables with `pedot-sim validate`. Thickness uses `time,mean_h` [s,m]; profiles use `x,h` [m,m]. Explicit current data use `t,I` [s,A]; monotonic CV branches use `V,I` [V,A]. Native molar-flux output cannot be compared to amperes. No extrapolation, nonfinite rows or ambiguous CV branch interpolation is accepted.
5. Set acceptance thresholds in advance, in observable units. Reports retain RMSE, mean absolute error, signed bias, R² where defined, point count, source metadata and SHA-256 hashes. A dataset-specific pass is not validation of other conditions or a guarantee of flawless manufacturing predictions.

ANOVA is restricted to complete, independently seeded stochastic factorials with estimable residual variance. Deterministic reruns and continuous sweeps receive descriptive analysis with explicit reasons for skipping ANOVA. Linear surrogates use standardized actual numeric inputs and leave-one-condition-out error; this measures interpolation of simulation outputs, not empirical accuracy. Candidate rankings have arbitrary weights and must not be presented as validated process optima.

## Measured manufacturing accounts

`analyze-manufacturing` accepts explicitly dimensioned `time_s,voltage_V,current_A` traces with a source identifier. It integrates the **product of linearly interpolated V and I exactly**, including zero crossings. Multiplying average V and average I, or treating the trapezoid of sampled power as exact, would give a different result for varying signals. Imported and returned terminal energy and positive/negative charge are reported separately. Samples cannot reconstruct unobserved waveform edges; the sampling diagnostic is a warning, not an accuracy certificate.

Optional measured dry product mass gives specific electrical energy. Product/initial-monomer mass ratio is **not** chemical yield: dopant, water, other material and collection losses affect it. Film area/thickness with mass gives effective density. Four-terminal resistance R with voltage-probe spacing L, conducting width w and thickness h gives `Rs=R*w/L` and `sigma=1/(Rs*h)` **only for a uniform rectangular path**. These relations do not apply directly to an irregular dendritic network. No model morphology is silently converted to conductivity, energy, yield or density. Instrument uncertainty is retained in metadata if supplied but is not propagated by this release. Raw input SHA-256 hashes accompany all accounts; auxiliary energy and lifecycle impacts remain outside the boundary.

## Ensemble interpretation

`run-dendrite-batch` records a period/seed plan and source/dependency fingerprint before execution. Resume requires identical parameters, code, environment and complete outputs. Failed/interrupted outputs remain inspectable and require a new campaign ID. Periods remain in model units; statistical replication does not create a physical frequency calibration.

`bridging_summary.csv` includes a 95% Wilson interval for the model's probability of bridging by the planned horizon, the number censored, the restricted mean `mean(min(T,H))`, and a separately labelled conditional mean among runs that bridge. Unbridged observations must reach the common horizon H. Restricted mean time is not mean completion time; early termination cannot masquerade as a completed unbridged observation. Seed variability quantifies this model's randomness, not experimental uncertainty or parameter uncertainty.


## Evidence depictions

[Data and visualization quality](DATA_AND_VISUALIZATION.md) defines the interpretation of each export. Published measurements, image-derived estimates, model predictions and illustrative geometry retain distinct labels. Model seed intervals describe stochastic variability only. Source spatial SD is not experimental repeatability. Potential surfaces encode voltage, not elevation; thickness histories encode time, not an additional spatial direction. A 2D projected area does not establish physical volume, dry mass or 3D fractal dimension.
