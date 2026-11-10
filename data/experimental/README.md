# Experimental comparison

No experimental measurements are included.

The Python function `pedot_sim.validation.run_validation` accepts simulation and experimental CSV paths. Both must contain t [s], I [A] for CA or V [V], I [A] for individually aligned, monotonic CV branches. Simulation `time` is accepted as an alias for t. Native molar `flux` is deliberately not renamed to current: a justified charge/geometry/observable conversion is needed first. CV cycles must be split into branches; extrapolation outside the simulated range is rejected.

The returned report contains `status: failed` and an explanation on failure. There is no `pedot-sim validate` CLI command. Agreement with a toy or synthetic CSV is not experimental validation.
