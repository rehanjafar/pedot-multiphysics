"""
Stochastic nucleation model for PEDOT film formation.

Based on the liquid-to-solid phase transformation observed by
Subramanian and Martin (Macromolecules 2021; Nano Lett. 2021):
- Liquid oligomer droplets form at the electrode surface
- Droplets nucleate, diffuse, and merge into polymer nodules
- Modeled as Poisson-distributed nucleation events with small
  thickness perturbations

All randomness uses numpy.random.default_rng(seed) for determinism.
"""

from __future__ import annotations

import numpy as np

from .config import NucleationConfig


class NucleationModel:
    """Stochastic nucleation for PEDOT film growth.

    Parameters
    ----------
    config : NucleationConfig
        Nucleation settings.
    n_electrode : int
        Number of grid cells along the electrode surface.
    """

    def __init__(self, config: NucleationConfig, n_electrode: int) -> None:
        self.config = config
        self.n_electrode = n_electrode
        self.rng = np.random.default_rng(config.seed)
        self.enabled = config.enable_noise
        self.events = np.zeros(n_electrode, dtype=int)

    def generate_initial_nuclei(self) -> np.ndarray:
        """Generate initial thickness perturbation on electrode surface.

        Returns
        -------
        h0 : np.ndarray, shape (n_electrode,)
            Initial film thickness [m] with small random perturbation.
        """
        if not self.enabled:
            return np.zeros(self.n_electrode)

        amp = self.config.noise_amplitude
        # Small uniform perturbation
        h0 = amp * self.rng.uniform(0, 1, size=self.n_electrode)
        return np.zeros(self.n_electrode)  # nuclei do not supply polymer mass

    def add_nucleation_noise(
        self,
        h: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Add stochastic nucleation events to film thickness.

        Models Poisson-distributed nucleation: in each cell,
        the probability of a nucleation event in dt is:
            p = rate * dt

        Each event adds a small thickness increment.

        Parameters
        ----------
        h : np.ndarray, shape (n_electrode,)
            Current thickness [m].
        dt : float
            Time step [s].

        Returns
        -------
        h_new : np.ndarray
        """
        if not self.enabled:
            return h.copy()

        raise ValueError("Additive film noise is unsupported; use flux_multiplier")

    def flux_multiplier(self, dt: float, dx: float) -> np.ndarray:
        """Dimensionless heterogeneity; deposition still consumes monomer.

        A phenomenological multiplicative model, not calibrated nucleation physics.
        """
        if not self.enabled:
            return np.ones(self.n_electrode)
        events = self.rng.poisson(self.config.nucleation_rate * dx * dt,
                                  size=self.n_electrode)
        self.events += events
        return 1. + self.events
