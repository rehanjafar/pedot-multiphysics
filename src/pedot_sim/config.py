"""
Configuration system using dataclasses with JSON serialization.

All physical quantities use SI units unless otherwise noted.
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from . import constants as C


@dataclass
class GeometryConfig:
    """Two-compartment salt-bridge domain geometry.

    Dimensions in meters.
    """
    compartment_width: float = C.COMPARTMENT_WIDTH
    compartment_height: float = C.COMPARTMENT_HEIGHT
    bridge_width: float = C.BRIDGE_WIDTH_NARROW
    bridge_length: float = C.BRIDGE_LENGTH
    electrode_length: float = C.ELECTRODE_LENGTH
    electrode_y_fraction: float = 0.5  # BPE position along bridge (0=left edge, 1=right edge)
    geometry_type: str = "narrow"  # "narrow" or "wide"

    def __post_init__(self) -> None:
        if self.geometry_type == "narrow":
            self.bridge_width = C.BRIDGE_WIDTH_NARROW
        elif self.geometry_type == "wide":
            self.bridge_width = C.BRIDGE_WIDTH_WIDE


@dataclass
class WaveformConfig:
    """Applied waveform parameters.

    V(t) = V_DC + V_AC * sin(2*pi*f*t)
    Left feeder: +V(t)/2, Right feeder: -V(t)/2
    """
    V_DC: float = 2.0        # DC offset [V]
    V_AC: float = 1.0        # AC amplitude [V]
    frequency: float = 10.0  # Frequency [Hz]
    waveform_type: str = "sinusoidal"  # sinusoidal / pulsed / asymmetric_adc
    duty_cycle: float = 0.5  # For pulsed waveform
    adc_ratio: float = 0.7   # For asymmetric ADC waveform
    rest_fraction: float = 0.0  # Fraction of period spent at rest


@dataclass
class PhysicsConfig:
    """Physics model settings."""
    physics_stage: str = "A"  # A / B / C / D
    diffusion_coefficient: float = C.D_EDOT_DEFAULT  # [m^2/s]
    viscosity: float = 0.003  # [Pa*s]
    conductivity: float = C.SIGMA_ELECTROLYTE  # [S/m]
    ionic_strength: float = 0.1  # [mol/L] (affects conductivity)
    C_bulk: float = C.C_BULK_DEFAULT  # [mol/m^3]
    k_rxn: float = C.K_RXN_DEFAULT  # [m/s]
    j0: float = C.J0_DEFAULT  # [A/m^2]
    alpha: float = C.ALPHA_DEFAULT
    n_electrons: int = C.N_ELECTRONS
    beta_field: float = C.BETA_FIELD
    z_species: int = C.Z_EDOT
    T: float = C.T_DEFAULT  # [K]


@dataclass
class NumericalConfig:
    """Numerical solver settings."""
    Nx: int = 100  # Grid points in x
    Ny: int = 100  # Grid points in y
    dt: float = 0.01  # Time step [s]
    t_end: float = 100.0  # Total simulation time [s]
    n_snapshots: int = 20  # Number of field snapshots to save
    implicit_theta: float = 0.5  # 0=explicit, 0.5=Crank-Nicolson, 1=fully implicit
    laplace_tol: float = 1.0e-6  # Laplace solver tolerance
    laplace_max_iter: int = 10000  # Max iterations for Laplace solver
    adaptive_dt: bool = False  # Enable adaptive time-stepping
    cfl_safety: float = 0.8  # CFL safety factor


@dataclass
class NucleationConfig:
    """Stochastic nucleation settings."""
    enable_noise: bool = True
    noise_amplitude: float = 1.0e-9  # [m] initial perturbation amplitude
    nucleation_rate: float = 1.0e3  # Poisson rate parameter [events/(m*s)]
    seed: int = 42


@dataclass
class OutputConfig:
    """Output and visualization settings."""
    save_fields: bool = True
    save_animations: bool = True
    dpi: int = 300
    animation_fps: int = 20
    figure_format: str = "png"  # png / svg


@dataclass
class BatchConfig:
    """Batch campaign settings."""
    campaign_id: str = "default"
    sweep_type: str = "factorial"  # factorial / grid / random / lhs
    frequencies: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    D_levels: list[str] = field(default_factory=lambda: ["low", "mid", "high"])
    geometries: list[str] = field(default_factory=lambda: ["narrow", "wide"])
    n_replicates: int = 3
    max_workers: int = 4
    n_dense_samples: int = 500
    # Dense sweep ranges
    freq_range: list[float] = field(default_factory=lambda: [0.5, 200.0])
    D_range: list[float] = field(default_factory=lambda: [5e-11, 5e-9])
    V_DC_range: list[float] = field(default_factory=lambda: [0.5, 4.0])
    V_AC_range: list[float] = field(default_factory=lambda: [0.2, 3.0])


@dataclass
class SimConfig:
    """Master simulation configuration."""
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    waveform: WaveformConfig = field(default_factory=WaveformConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    nucleation: NucleationConfig = field(default_factory=NucleationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)

    def copy(self) -> SimConfig:
        """Return a deep copy of this config."""
        return copy.deepcopy(self)


def _to_serializable(obj: Any) -> Any:
    """Convert dataclass fields to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_config(config: SimConfig, path: Path) -> None:
    """Save configuration to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _to_serializable(asdict(config))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_config(path: Path) -> SimConfig:
    """Load configuration from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    geo = GeometryConfig(**data.get("geometry", {}))
    wav = WaveformConfig(**data.get("waveform", {}))
    phy = PhysicsConfig(**data.get("physics", {}))
    num = NumericalConfig(**data.get("numerical", {}))
    nuc = NucleationConfig(**data.get("nucleation", {}))
    out = OutputConfig(**data.get("output", {}))

    batch_data = data.get("batch", {})
    bat = BatchConfig(**batch_data)

    return SimConfig(
        geometry=geo, waveform=wav, physics=phy,
        numerical=num, nucleation=nuc, output=out, batch=bat,
    )
