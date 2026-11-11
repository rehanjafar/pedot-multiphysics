# Experimental data

No measured datasets are included. Synthetic test fixtures are only software tests.

For `pedot-sim validate`, provide a CSV and JSON metadata with a source identifier
and exact SI unit declarations. Thickness: `time,mean_h` [s,m]; profile: `x,h` [m,m];
CA: `t,I` [s,A]; CV: `V,I` [V,A] after separating monotonic scan branches.
Record instrument, chemistry, electrode geometry, waveform convention, sample
identity, replicate, uncertainty and calibration/validation partition in metadata.

The directly driven wire model currently reports lattice cells and model steps.
Do not label those quantities metres or seconds without a calibrated mapping.
For morphology validation, retain original scaled micrographs/time sequences and
use the same segmentation for simulated and measured projected structures.

See ../../docs/SCIENTIFIC_BASIS.md for the evidence and validation protocol.
