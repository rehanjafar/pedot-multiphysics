"""
Physical constants and PEDOT-specific parameters.

Sources:
- CODATA 2018 for fundamental constants
- Material/kinetic presets below are exploratory, uncalibrated values.
  No quantitative literature provenance or experimental fit is claimed.

Units are SI unless otherwise noted.
"""

from dataclasses import dataclass
import math

# ---------------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# ---------------------------------------------------------------------------
F: float = 96485.3321233100184  # N_A * e, exact SI defining constants [C/mol]
R_GAS: float = 8.31446261815324  # N_A * k_B, exact SI [J/(mol*K)]
k_B: float = 1.380649e-23  # Boltzmann constant [J/K]
N_A: float = 6.02214076e23  # Avogadro number [1/mol]
e_charge: float = 1.602176634e-19  # Elementary charge [C]

# ---------------------------------------------------------------------------
# Default temperature
# ---------------------------------------------------------------------------
T_DEFAULT: float = 298.15  # Standard temperature [K]

# ---------------------------------------------------------------------------
# EDOT monomer properties
# ---------------------------------------------------------------------------
M_EDOT: float = 142.18e-3  # Molar mass of EDOT [kg/mol]
M_PEDOT_REPEAT: float = 140.16e-3  # Neutral C6H4O2S repeat, excludes dopant [kg/mol]
RHO_PEDOT: float = 1340.0  # Approximate PEDOT film density [kg/m^3]
R_EDOT: float = 3.5e-10  # Hydrodynamic radius of EDOT molecule [m]

# ---------------------------------------------------------------------------
# Default electrochemical parameters
# ---------------------------------------------------------------------------
C_BULK_DEFAULT: float = 50.0  # Bulk EDOT concentration [mol/m^3] (0.05 M)
D_EDOT_DEFAULT: float = 6.5e-10  # Exploratory diffusion coefficient [m^2/s]
Z_EDOT: int = 0  # Neutral EDOT monomer; charged intermediates need a separate species model
N_ELECTRONS: int = 2  # Electrons transferred per EDOT oxidation
SIGMA_ELECTROLYTE: float = 1.0  # Electrolyte conductivity [S/m]

# ---------------------------------------------------------------------------
# Kinetic parameters
# ---------------------------------------------------------------------------
K_RXN_DEFAULT: float = 1.0e-3  # Heterogeneous rate constant [m/s]
J0_DEFAULT: float = 10.0  # Exchange current density [A/m^2]
ALPHA_DEFAULT: float = 0.5  # Charge transfer coefficient (symmetric)
E0_FORMAL: float = 0.0  # Formal potential for EDOT oxidation [V vs ref]

# ---------------------------------------------------------------------------
# Field enhancement factor (Stage B)
# ---------------------------------------------------------------------------
BETA_FIELD: float = 0.01  # Field enhancement coefficient [m/V]

# ---------------------------------------------------------------------------
# Viscosity levels mapped to diffusion coefficients
# ---------------------------------------------------------------------------
# Stokes-Einstein: D = k_B * T / (6 * pi * mu * r)
# With r = R_EDOT = 3.5 Angstrom, T = 298.15 K

def stokes_einstein_D(mu: float, T: float = T_DEFAULT, r: float = R_EDOT) -> float:
    """
    Compute diffusion coefficient from Stokes-Einstein relation.

    D = k_B * T / (6 * pi * mu * r)

    Parameters
    ----------
    mu : float
        Dynamic viscosity [Pa*s].
    T : float
        Temperature [K].
    r : float
        Hydrodynamic radius [m].

    Returns
    -------
    float
        Diffusion coefficient [m^2/s].
    """
    if not all(math.isfinite(v) and v > 0 for v in (mu, T, r)):
        raise ValueError("Viscosity, temperature and hydrodynamic radius must be positive and finite")
    return k_B * T / (6.0 * math.pi * mu * r)


@dataclass(frozen=True)
class ViscosityLevel:
    """Pre-defined viscosity levels for factorial design."""
    name: str
    mu: float  # [Pa*s]
    D: float   # [m^2/s]


# Legacy heuristic levels, NOT derived from the stated Stokes-Einstein inputs.
# Labels and pairs below define a sensitivity design, not measured solvent properties.
VISCOSITY_LOW = ViscosityLevel("low", 0.001, 2.0e-9)
VISCOSITY_MID = ViscosityLevel("mid", 0.003, 6.5e-10)
VISCOSITY_HIGH = ViscosityLevel("high", 0.01, 1.0e-10)

VISCOSITY_LEVELS = {
    "low": VISCOSITY_LOW,
    "mid": VISCOSITY_MID,
    "high": VISCOSITY_HIGH,
}

D_LEVELS = {
    "low": VISCOSITY_HIGH.D,   # High viscosity -> low D
    "mid": VISCOSITY_MID.D,
    "high": VISCOSITY_LOW.D,   # Low viscosity -> high D
}

# ---------------------------------------------------------------------------
# Geometry defaults [m]
# ---------------------------------------------------------------------------
COMPARTMENT_WIDTH: float = 0.01  # 1 cm
COMPARTMENT_HEIGHT: float = 0.01  # 1 cm
BRIDGE_LENGTH: float = 0.005  # 5 mm
BRIDGE_WIDTH_NARROW: float = 0.002  # 2 mm
BRIDGE_WIDTH_WIDE: float = 0.005  # 5 mm
ELECTRODE_LENGTH: float = 0.003  # 3 mm (BPE length along bridge)
