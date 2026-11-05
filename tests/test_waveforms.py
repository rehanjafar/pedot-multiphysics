"""Tests for the waveform module."""

import numpy as np
import pytest

from pedot_sim.config import WaveformConfig
from pedot_sim.waveforms import (
    asymmetric_adc,
    get_boundary_potentials,
    get_waveform,
    pulsed,
    sinusoidal,
    with_rest,
)


class TestSinusoidal:
    def test_amplitude(self):
        """Peak value should be V_DC + V_AC."""
        t = np.linspace(0, 1, 10000)
        V = sinusoidal(t, V_DC=2.0, V_AC=1.0, freq=10.0)
        assert np.max(V) == pytest.approx(3.0, abs=0.01)

    def test_minimum(self):
        """Minimum should be V_DC - V_AC."""
        t = np.linspace(0, 1, 10000)
        V = sinusoidal(t, V_DC=2.0, V_AC=1.0, freq=10.0)
        assert np.min(V) == pytest.approx(1.0, abs=0.01)

    def test_frequency(self):
        """Should complete correct number of cycles."""
        freq = 5.0
        t = np.linspace(0, 1, 100000)
        V = sinusoidal(t, V_DC=0, V_AC=1.0, freq=freq)
        # Count zero crossings (rising)
        crossings = np.sum((V[:-1] < 0) & (V[1:] >= 0))
        assert crossings == pytest.approx(freq, abs=1)

    def test_dc_offset(self):
        """Mean over full cycles should approximate V_DC."""
        t = np.linspace(0, 10, 100000)  # 10 full cycles at 1 Hz
        V = sinusoidal(t, V_DC=3.0, V_AC=1.0, freq=1.0)
        assert np.mean(V) == pytest.approx(3.0, abs=0.01)

    def test_at_zero(self):
        """V(0) = V_DC for sinusoidal."""
        assert sinusoidal(0.0, V_DC=2.0, V_AC=1.0, freq=10.0) == pytest.approx(2.0)


class TestPulsed:
    def test_on_phase(self):
        """During on phase, V = V_DC + V_AC."""
        # At t=0, phase=0 < duty_cycle=0.5, so should be on
        V = pulsed(0.0, V_DC=2.0, V_AC=1.0, freq=1.0, duty_cycle=0.5)
        assert V == pytest.approx(3.0)

    def test_off_phase(self):
        """During off phase, V = V_DC - V_AC."""
        # At t=0.75, phase=0.75 > duty_cycle=0.5, so should be off
        V = pulsed(0.75, V_DC=2.0, V_AC=1.0, freq=1.0, duty_cycle=0.5)
        assert V == pytest.approx(1.0)

    def test_array_input(self):
        t = np.array([0.0, 0.25, 0.5, 0.75])
        V = pulsed(t, V_DC=2.0, V_AC=1.0, freq=1.0, duty_cycle=0.5)
        assert len(V) == 4


class TestBoundaryPotentials:
    def test_symmetric(self):
        V_left, V_right = get_boundary_potentials(4.0)
        assert V_left == pytest.approx(2.0)
        assert V_right == pytest.approx(-2.0)

    def test_zero(self):
        V_left, V_right = get_boundary_potentials(0.0)
        assert V_left == pytest.approx(0.0)
        assert V_right == pytest.approx(0.0)


class TestGetWaveform:
    def test_sinusoidal_config(self):
        config = WaveformConfig(V_DC=2.0, V_AC=1.0, frequency=10.0, waveform_type="sinusoidal")
        wf = get_waveform(config)
        assert callable(wf)
        assert wf(0.0) == pytest.approx(2.0)

    def test_pulsed_config(self):
        config = WaveformConfig(V_DC=2.0, V_AC=1.0, frequency=1.0, waveform_type="pulsed")
        wf = get_waveform(config)
        assert wf(0.0) == pytest.approx(3.0)


class TestWithRest:
    def test_active_phase(self):
        """During active phase, should return sinusoidal value."""
        V = with_rest(0.0, V_DC=2.0, V_AC=1.0, freq=1.0, rest_fraction=0.2)
        assert V == pytest.approx(2.0)

    def test_rest_phase(self):
        """During rest phase, should return 0."""
        # At phase = 0.9 (> 1 - 0.2 = 0.8), should be in rest
        V = with_rest(0.9, V_DC=2.0, V_AC=1.0, freq=1.0, rest_fraction=0.2)
        assert V == pytest.approx(0.0)
