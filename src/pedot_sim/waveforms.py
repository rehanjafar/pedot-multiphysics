"""
Waveform library for applied potential signals.

V(t) = V_DC + V_AC * sin(2*pi*f*t)   [sinusoidal]

Feeder boundary conditions:
  Left feeder:  +V(t)/2
  Right feeder: -V(t)/2
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .config import WaveformConfig


def sinusoidal(t: float | np.ndarray, V_DC: float, V_AC: float, freq: float) -> float | np.ndarray:
    """Sinusoidal waveform.

    V(t) = V_DC + V_AC * sin(2*pi*f*t)
    """
    return V_DC + V_AC * np.sin(2.0 * np.pi * freq * t)


def pulsed(
    t: float | np.ndarray,
    V_DC: float,
    V_AC: float,
    freq: float,
    duty_cycle: float = 0.5,
) -> float | np.ndarray:
    """Pulsed (square) waveform.

    V(t) = V_DC + V_AC  during on-phase
    V(t) = V_DC - V_AC  during off-phase
    """
    phase = (t * freq) % 1.0
    if isinstance(phase, np.ndarray):
        result = np.where(phase < duty_cycle, V_DC + V_AC, V_DC - V_AC)
    else:
        result = V_DC + V_AC if phase < duty_cycle else V_DC - V_AC
    return result


def asymmetric_adc(
    t: float | np.ndarray,
    V_DC: float,
    V_AC: float,
    freq: float,
    adc_ratio: float = 0.7,
) -> float | np.ndarray:
    """Asymmetric AC/DC waveform (Lee et al. 2024).

    Positive half-cycle is scaled by adc_ratio,
    negative half-cycle scaled by (1 - adc_ratio).
    """
    phase = (t * freq) % 1.0
    sin_val = np.sin(2.0 * np.pi * freq * t)
    if isinstance(sin_val, np.ndarray):
        scale = np.where(sin_val >= 0, adc_ratio, 1.0 - adc_ratio)
    else:
        scale = adc_ratio if sin_val >= 0 else 1.0 - adc_ratio
    return V_DC + V_AC * scale * np.sin(2.0 * np.pi * freq * t)


def with_rest(
    t: float | np.ndarray,
    V_DC: float,
    V_AC: float,
    freq: float,
    rest_fraction: float = 0.2,
) -> float | np.ndarray:
    """Sinusoidal with rest intervals.

    For the last `rest_fraction` of each period, V(t) = 0.
    """
    phase = (t * freq) % 1.0
    active = sinusoidal(t, V_DC, V_AC, freq)
    if isinstance(phase, np.ndarray):
        return np.where(phase < (1.0 - rest_fraction), active, 0.0)
    return active if phase < (1.0 - rest_fraction) else 0.0


def get_waveform(config: WaveformConfig) -> Callable[[float | np.ndarray], float | np.ndarray]:
    """Return a waveform function V(t) based on config.

    Parameters
    ----------
    config : WaveformConfig

    Returns
    -------
    Callable
        Function V(t) -> applied potential [V].
    """
    wtype = config.waveform_type.lower()

    def waveform_func(t: float | np.ndarray) -> float | np.ndarray:
        if wtype == "sinusoidal":
            if config.rest_fraction > 0:
                return with_rest(t, config.V_DC, config.V_AC, config.frequency, config.rest_fraction)
            return sinusoidal(t, config.V_DC, config.V_AC, config.frequency)
        elif wtype == "pulsed":
            return pulsed(t, config.V_DC, config.V_AC, config.frequency, config.duty_cycle)
        elif wtype == "asymmetric_adc":
            return asymmetric_adc(t, config.V_DC, config.V_AC, config.frequency, config.adc_ratio)
        else:
            return sinusoidal(t, config.V_DC, config.V_AC, config.frequency)

    return waveform_func


def get_boundary_potentials(V_total: float) -> tuple[float, float]:
    """Compute feeder electrode boundary potentials.

    Parameters
    ----------
    V_total : float
        Total applied potential V(t) [V].

    Returns
    -------
    (V_left, V_right) : tuple[float, float]
        Left feeder = +V/2, Right feeder = -V/2.
    """
    return (+V_total / 2.0, -V_total / 2.0)
