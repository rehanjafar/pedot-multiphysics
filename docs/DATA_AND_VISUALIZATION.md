# Data and visualization quality

Quality means a defensible observation or calculation, explicit units and provenance, and a depiction that answers a specific question. More digits or an attractive 3D surface do not establish physical accuracy. This release does not manufacture precision, uncertainty bands, micrographs or missing measurements.

## Published reference evidence

The [Janzakova et al. paper](https://doi.org/10.1038/s41467-021-27274-9) provides the scientific context for the [authors' version-1 numerical dataset](https://doi.org/10.6084/m9.figshare.16814710.v1). The selected tables retain 2,138 records, not 2,138 independent experiments. Data are CC BY 4.0; [full attribution and transformations](../src/pedot_sim/datasets/janzakova2021/ATTRIBUTION.md) accompany the package and atlas.

| Table | Records | Purpose and interpretation |
|---|---:|---|
| Amplitude morphology, Fig. 2h | 7 | Area, endpoint time, image fractal number, six position values and spatial SD; 80 Hz, 50% duty, zero offset. |
| Frequency morphology, Fig. 4b | 12 | Same observables at 5 V peak, 50% duty, zero offset. The 10 Hz area is reported as zero while other morphology measures are nonzero; preserve and flag it. |
| Offset morphology, Fig. 3d | 10 | Separate wire areas and their derived signed imbalance; 5 V peak, 80 Hz, 50% duty. |
| Growth traces, Fig. 2f | 672 | Actual nonblank time/area samples in seven amplitude series; retain initial area, sample order and differing endpoints. |
| Impedance, Fig. 4c | 183 | Three sample IDs, 61 frequencies each, measured after growth in 1 mM NaPSS without EDOT/BQ; retain modulus/phase and derive Cartesian impedance without fitting. Probe frequency differs from growth frequency. |
| Voltammetry, Fig. 2a | 1,254 | First cycle in three separate solutions, in acquisition order; potential is relative to Ag/AgCl, not two-wire drive voltage. |

The raw file also contains compact circuit simulations and fitted parameters. They are not included in the measurement tables. No fits, smoothing, imputation, baseline correction or inferred sample identities are introduced. Source row and column indices are one based. Area columns are converted from mm² to m²; plots convert back explicitly. Recorded decimal precision is preserved without implying measurement uncertainty at that precision.

**Uncertainty:** branching error bars are the sample SD across six image positions within one morphology. They are not standard errors or confidence intervals across independent growth experiments. Instrument, segmentation and sample-to-sample uncertainties are not recoverable from these selected values. The zero-area anomaly remains unresolved and is not corrected, discarded silently or used to fit a logarithmic trend.

The measured records are not a calibration of our lattice model. Physical length/time mapping, matched apparatus/chemistry and independent experiments remain necessary. The [chemistry study by Baron et al.](https://doi.org/10.1039/d5lp00225g) further limits transfer across electrolytes and redox resources; ideal equipotential deposits omit the polymer resistance considered there.

## Reproduction

```sh
pedot-sim analyze-literature --output-dir data/published_evidence
# Optional: reproduce bundled subsets from the downloaded, exact original file.
pedot-sim curate-literature --source-csv dataset.csv --output-dir data/recurated
```

Original CSV: https://ndownloader.figshare.com/files/31090411

SHA-256: `3a6dac6f58120cf5c05469952846da6718f72ddaac8cbc049d5cbec1070129fc`

The importer checks this hash before decoding the semicolon-delimited Windows-1252 file or writing output. An unreviewed revision fails closed. `source_metadata.json` records hashes of every bundled subset; installed-wheel loading checks them again. The exported gallery copies those CSVs byte for byte. This verifies data integrity, not the truth of every reported observation.

## Representations and their purpose

| Output | Purpose | Limit |
|---|---|---|
| Published growth in 3D parameter space | Compare actual time/voltage/area histories | Lines join observations within a series; no interpolated surface between conditions and no spatial reconstruction. |
| Direct-wire initial/final maps and time sequence | Locate attachment and see recorded development | Lattice cells and model steps; no physical scale is inferred. |
| Potential and field maps | See the electrical environment driving the last interval | Saved potential belongs to the last solved pre-attachment geometry at its midpoint, never a shorted final geometry. Field derivatives are V/cell. |
| Network OBJ and interactive mesh | Inspect network arrangement | One-cell assumed extrusion; no measured thickness, volume, printability or 3D transport. Diagonal contacts can create touching surface edges. |
| Fixed-strip concentration views | Resolve the local depletion layer and domain context | Actual graded cell edges and shared concentration scale; this is the separate legacy model. |
| Thickness history surface and GIF | Inspect spatial growth and recorded timing | Time is a history axis. GIF playback is compressed; timestamps remain actual. A single profile uses an explicitly arbitrary display width. |
| Every-step inventory chart | Audit numerical material conservation | Independent reservoir-input calculation versus consumed and remaining inventory, all per metre depth. |
| Seed ensemble | Show every stochastic result and bridge probability | Wilson intervals quantify model randomness at the planned horizon, not experimental uncertainty. |
| Measured-trace electrical accounts | Inspect supplied V/I, power samples and cumulative energy | Exact integral of the product of linearly interpolated signals; no invented uncertainty or total lifecycle claim. |

Static charts export PNG and editable SVG; interactive views export offline HTML and Plotly JSON. Captions, origins, download links and interpretation limits accompany each gallery. Raw CSV/NPZ values remain authoritative. Plot surfaces connect saved values and do not recover unrecorded transients. A failed required gallery makes a new run fail instead of silently omitting requested depictions.

Open `views/index.html` within each run, or regenerate it with `pedot-sim visualize-run --run-dir PATH`. Keep the gallery and its local JavaScript bundle together. Existing numerical files are left intact. The published gallery is `OUTPUT_DIR/index.html`; ensemble views are `dendrites/CAMPAIGN/views/index.html`.
