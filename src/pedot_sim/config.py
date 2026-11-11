"""
Configuration system using dataclasses with JSON serialization.

All physical quantities use SI units unless otherwise noted.
"""

from __future__ import annotations

import json
import copy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

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
        if self.bridge_width <= 0:
            raise ValueError("bridge_width must be positive; geometry_type is a label only")


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
    ionic_strength: float = 0.1  # [mol/L] metadata; conductivity is independently specified
    C_bulk: float = C.C_BULK_DEFAULT  # [mol/m^3]
    k_rxn: float = C.K_RXN_DEFAULT  # [m/s]
    j0: float = C.J0_DEFAULT  # [A/m^2]
    alpha: float = C.ALPHA_DEFAULT
    n_electrons: int = C.N_ELECTRONS
    beta_field: float = C.BETA_FIELD
    z_species: int = C.Z_EDOT
    T: float = C.T_DEFAULT  # [K]
    deposited_molar_mass: float = C.M_PEDOT_REPEAT  # dry neutral repeat [kg/mol]
    film_density: float = C.RHO_PEDOT  # assumed effective film density [kg/m^3]
    deposition_efficiency: float = 1.0  # fraction of consumed monomer retained in film


@dataclass
class NumericalConfig:
    """Numerical solver settings."""
    Nx: int = 100  # Grid points in x
    Ny: int = 100  # Grid points in y
    dt: float = 0.01  # Time step [s]
    t_end: float = 100.0  # Total simulation time [s]
    n_snapshots: int = 20  # Number of field snapshots to save
    implicit_theta: float = 1.0  # positivity-preserving backward Euler by default
    laplace_tol: float = 1.0e-6  # Laplace solver tolerance
    laplace_max_iter: int = 10000  # Max iterations for Laplace solver
    mesh_stretch: float = 0.0  # >0 enables boundary-fitted sinh grading
    adaptive_dt: bool = False  # Unsupported; True is explicitly rejected
    cfl_safety: float = 0.8  # Migration substep probability/CFL limit


@dataclass
class NucleationConfig:
    """Stochastic nucleation settings."""
    enable_noise: bool = False
    noise_amplitude: float = 1.0e-9  # [m] initial perturbation amplitude
    nucleation_rate: float = 1.0e3  # Poisson rate parameter [events/(m*s)]
    seed: int = 42


@dataclass
class OutputConfig:
    """Output and visualization settings."""
    save_fields: bool = True
    save_animations: bool = False
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


def validate_config(config: SimConfig) -> None:
    """Reject invalid or silently unsupported inputs before writing any output."""
    import math
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(v) for v in value)
        return not isinstance(value, (int, float)) or math.isfinite(value)
    if not finite(asdict(config)):
        raise ValueError("Configuration numbers must be finite")
    n, g, ph, w = config.numerical, config.geometry, config.physics, config.waveform
    for name, value, minimum in (
        ("Nx", n.Nx, 3), ("Ny", n.Ny, 3), ("n_snapshots", n.n_snapshots, 0),
        ("laplace_max_iter", n.laplace_max_iter, 1),
        ("seed", config.nucleation.seed, 0), ("n_electrons", ph.n_electrons, 1),
        ("dpi", config.output.dpi, 1), ("animation_fps", config.output.animation_fps, 1),
        ("n_replicates", config.batch.n_replicates, 1),
        ("max_workers", config.batch.max_workers, 1),
        ("n_dense_samples", config.batch.n_dense_samples, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(ph.z_species) is not int:
        raise ValueError("z_species must be an integer charge number")
    for name, value in (("enable_noise", config.nucleation.enable_noise),
                        ("save_fields", config.output.save_fields),
                        ("save_animations", config.output.save_animations),
                        ("adaptive_dt", n.adaptive_dt)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if not 0 <= n.mesh_stretch <= 12:
        raise ValueError("mesh_stretch must be between 0 and 12")
    if n.adaptive_dt:
        raise ValueError("adaptive_dt is not implemented; use timestep refinement")
    if n.dt <= 0 or n.t_end <= 0 or not 0 <= n.implicit_theta <= 1:
        raise ValueError("Invalid timestep, duration or theta")
    if n.laplace_tol <= 0 or not 0 < n.cfl_safety <= 1:
        raise ValueError("Invalid solver settings")
    if min(g.compartment_width, g.compartment_height, g.bridge_length,
           g.bridge_width, g.electrode_length) <= 0:
        raise ValueError("Geometry dimensions must be positive")
    if g.bridge_width > g.compartment_height or g.electrode_length > g.bridge_length:
        raise ValueError("Bridge/electrode does not fit domain")
    if g.electrode_y_fraction != 0.5:
        raise ValueError("Only a centred electrode is currently supported")
    if ph.physics_stage not in ("A", "B", "C", "D"):
        raise ValueError("Unknown physics stage; use A, B, C or D")
    if min(ph.C_bulk, ph.T, ph.viscosity, ph.conductivity,
           ph.deposited_molar_mass, ph.film_density) <= 0:
        raise ValueError("Concentration, temperature, viscosity, conductivity and material properties must be positive")
    if min(ph.diffusion_coefficient, ph.k_rxn, ph.j0, ph.beta_field, ph.ionic_strength) < 0:
        raise ValueError("Transport and kinetic parameters must be nonnegative")
    if not 0 < ph.alpha < 1 or not 0 <= ph.deposition_efficiency <= 1:
        raise ValueError("Invalid transfer coefficient or deposition efficiency")
    if w.frequency <= 0 or w.V_AC < 0 or w.waveform_type not in ("sinusoidal", "pulsed", "asymmetric_adc"):
        raise ValueError("Invalid waveform")
    if not 0 < w.duty_cycle < 1 or not 0 <= w.rest_fraction < 1 or not 0 <= w.adc_ratio <= 1:
        raise ValueError("Invalid waveform fractions")
    if config.nucleation.nucleation_rate < 0 or config.nucleation.noise_amplitude < 0:
        raise ValueError("Invalid stochastic settings")
    if config.output.figure_format not in ("png", "svg"):
        raise ValueError("Unsupported figure format")
    for name in ("freq_range", "D_range", "V_DC_range", "V_AC_range"):
        values = getattr(config.batch, name)
        if len(values) != 2 or values[0] >= values[1]:
            raise ValueError(f"{name} must contain two increasing bounds")
        if name in ("freq_range", "D_range") and values[0] <= 0:
            raise ValueError(f"{name} must be positive")
        if name == "V_AC_range" and values[0] < 0:
            raise ValueError("V_AC_range must be nonnegative")
    if not config.batch.frequencies or any(f <= 0 for f in config.batch.frequencies):
        raise ValueError("Batch frequencies must be positive and nonempty")
    if not config.batch.D_levels or not set(config.batch.D_levels) <= set(C.D_LEVELS):
        raise ValueError("Unknown or empty D_levels")
    if not config.batch.geometries or not set(config.batch.geometries) <= {"narrow", "wide"}:
        raise ValueError("Unknown or empty batch geometries")


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

    allowed = {"geometry", "waveform", "physics", "numerical", "nucleation", "output", "batch"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ValueError("Configuration contains unknown top-level sections")
    geo = GeometryConfig(**data.get("geometry", {}))
    wav = WaveformConfig(**data.get("waveform", {}))
    phy = PhysicsConfig(**data.get("physics", {}))
    num = NumericalConfig(**data.get("numerical", {}))
    nuc = NucleationConfig(**data.get("nucleation", {}))
    out = OutputConfig(**data.get("output", {}))

    batch_data = data.get("batch", {})
    bat = BatchConfig(**batch_data)

    config = SimConfig(
        geometry=geo, waveform=wav, physics=phy,
        numerical=num, nucleation=nuc, output=out, batch=bat,
    )
    validate_config(config)
    return config
