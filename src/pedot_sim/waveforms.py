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
    """Exploratory asymmetric half-sine waveform (not the Lee 2024 protocol).

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
            value = sinusoidal(t, config.V_DC, config.V_AC, config.frequency)
        elif wtype == "pulsed":
            value = pulsed(t, config.V_DC, config.V_AC, config.frequency, config.duty_cycle)
        elif wtype == "asymmetric_adc":
            value = asymmetric_adc(t, config.V_DC, config.V_AC, config.frequency, config.adc_ratio)
        else:
            raise ValueError(f"Unknown waveform: {wtype}")
        if config.rest_fraction > 0:
            value = np.where((np.asarray(t)*config.frequency) % 1 < 1-config.rest_fraction, value, 0.)
        return value

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


def integration_times(config):
    """Step endpoints including pulse/rest transitions; no step crosses a jump.

    Continuous signals receive >=40 samples per period. All waveforms are
    sampled at step midpoints. This resolves the drive, not electrochemistry.
    """
    w, n = config.waveform, config.numerical
    dt = n.dt if config.physics.physics_stage == "A" else min(n.dt, 1/(40*w.frequency))
    count = int(np.ceil(n.t_end / dt))
    if count > 2_000_000:
        raise ValueError("More than two million timesteps requested; shorten duration or lower frequency")
    endpoints = np.minimum(np.arange(count + 1) * dt, n.t_end)
    if config.physics.physics_stage != "A":
        phases = [0.]
        if w.waveform_type == "pulsed":
            phases.append(w.duty_cycle)
        if w.rest_fraction:
            phases.append(1-w.rest_fraction)
        cycles = np.arange(int(np.ceil(n.t_end*w.frequency))+1)
        jumps = np.concatenate([(cycles+phase)/w.frequency for phase in phases])
        endpoints = np.sort(np.concatenate([endpoints, jumps[(jumps>0)&(jumps<n.t_end)]]))
        # Merge roundoff duplicates without discarding the final endpoint.
        keep = np.r_[np.diff(endpoints) > 32*np.finfo(float).eps*max(1., n.t_end), True]
        endpoints = endpoints[keep]
    return endpoints
