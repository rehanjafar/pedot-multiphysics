"""Tests for the metrics module."""

import numpy as np
import pytest

from pedot_sim.metrics import (
    anisotropy_index,
    depletion_depth,
    mean_growth_rate,
    roughness_metric,
    timescale_ratio,
    total_charge_proxy,
    uniformity_index,
)
from pedot_sim.utils_grid import create_grid


class TestAnisotropyIndex:
    def test_uniform_gives_one(self):
        h = np.ones(50) * 1e-6
        assert anisotropy_index(h) == pytest.approx(1.0)

    def test_peaked_gives_greater_than_one(self):
        h = np.ones(50) * 1e-6
        h[25] = 5e-6  # one tall peak
        ai = anisotropy_index(h)
        assert ai > 1.0

    def test_empty_gives_one(self):
        h = np.array([])
        assert anisotropy_index(h) == 1.0

    def test_zero_gives_one(self):
        h = np.zeros(10)
        assert anisotropy_index(h) == 1.0

    def test_single_point(self):
        h = np.array([1e-6])
        assert anisotropy_index(h) == pytest.approx(1.0)


class TestUniformityIndex:
    def test_uniform_gives_zero(self):
        h = np.ones(50) * 1e-6
        assert uniformity_index(h) == pytest.approx(0.0, abs=1e-15)

    def test_varying_gives_positive(self):
        h = np.array([1e-6, 2e-6, 3e-6, 4e-6, 5e-6])
        ui = uniformity_index(h)
        assert ui > 0

    def test_empty_gives_zero(self):
        assert uniformity_index(np.array([])) == 0.0


class TestDepletionDepth:
    def test_no_depletion(self):
        """When C = C_bulk everywhere, depletion depth should be 0."""
        grid = create_grid(0.01, 0.01, 20, 20)
        C = np.ones((20, 20)) * 50.0
        electrode_mask = np.zeros((20, 20), dtype=bool)
        electrode_mask[10, 5:15] = True
        depth = depletion_depth(C, 50.0, grid, electrode_mask)
        assert depth == pytest.approx(0.0)

    def test_with_depletion(self):
        """When C < 0.9*C_bulk near electrode, depth should be positive."""
        grid = create_grid(0.01, 0.01, 20, 20)
        C = np.ones((20, 20)) * 50.0
        electrode_mask = np.zeros((20, 20), dtype=bool)
        electrode_mask[10, 5:15] = True
        # Create depletion zone near electrode
        C[9:12, :] = 40.0  # below 0.9 * 50 = 45
        depth = depletion_depth(C, 50.0, grid, electrode_mask)
        assert depth > 0


class TestTimescaleRatio:
    def test_computation(self):
        D = 6.5e-10  # m^2/s
        k = 1e-3  # m/s
        L = 3e-3  # m
        ratio = timescale_ratio(D, k, L)
        expected = L * k / D
        assert ratio == pytest.approx(expected)

    def test_zero_D(self):
        assert timescale_ratio(0, 1e-3, 1e-3) == float("inf")

    def test_diffusion_limited(self):
        """High L*k/D -> diffusion limited -> ratio > 1."""
        ratio = timescale_ratio(1e-10, 1e-2, 1e-2)
        assert ratio > 1

    def test_reaction_limited(self):
        """Low L*k/D -> reaction limited -> ratio < 1."""
        ratio = timescale_ratio(1e-8, 1e-6, 1e-3)
        assert ratio < 1


class TestMeanGrowthRate:
    def test_basic(self):
        t = np.array([0.0, 50.0, 100.0])
        h_history = np.array([
            [0.0, 0.0, 0.0],
            [1e-6, 1e-6, 1e-6],
            [2e-6, 2e-6, 2e-6],
        ])
        rate = mean_growth_rate(h_history, t)
        assert rate == pytest.approx(2e-6 / 100.0)

    def test_single_point(self):
        assert mean_growth_rate(np.array([[1e-6]]), np.array([0.0])) == 0.0


class TestTotalChargeProxy:
    def test_basic(self):
        flux = np.array([1.0, 2.0, 3.0])
        dt = np.array([0.1, 0.1, 0.1])
        assert total_charge_proxy(flux, dt) == pytest.approx(0.6)


class TestRoughnessMetric:
    def test_flat_is_zero(self):
        h = np.ones(50) * 1e-6
        assert roughness_metric(h, 1e-4) == pytest.approx(0.0)

    def test_rough_is_positive(self):
        h = np.array([0, 1e-6, 0, 1e-6, 0], dtype=float)
        assert roughness_metric(h, 1e-4) > 0
