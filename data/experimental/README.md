# Experimental Data Directory

Place experimental data files here for validation against simulation results.

## Accepted Formats

### Cyclic Voltammetry (CV) Data
- **File format**: CSV
- **Required columns**: `V` (potential in Volts), `I` (current in Amps)
- **Example filename**: `cv_pedot_10hz.csv`

```csv
V,I
-0.5,0.00001
-0.4,0.00002
...
```

### Chronoamperometry (CA) Data
- **File format**: CSV
- **Required columns**: `t` (time in seconds), `I` (current in Amps)
- **Example filename**: `ca_pedot_dc2v.csv`

```csv
t,I
0.0,0.001
0.1,0.00095
...
```

## Usage

Run validation with:
```bash
pedot-sim validate --sim-run data/simulated_raw/campaign_id/run_id --exp-data data/experimental/your_file.csv --type CA
```

## Notes
- Ensure consistent units (SI)
- Time series should start at t=0
- CV data should cover at least one full cycle
