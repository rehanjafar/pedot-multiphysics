"""
Electrochemical kinetics for PEDOT electropolymerization.

Butler-Volmer equation at the bipolar electrode:
    j = j0 * (C/C_ref) * [exp(alpha*n*F*eta/(R*T)) - exp(-(1-alpha)*n*F*eta/(R*T))]

Conversion to monomer consumption rate:
    R_boundary = max(j, 0) / (n * F)  [mol/(m^2*s)]

Conversion to film growth rate:
    dh/dt = max(j, 0) * M / (n * F * rho)  [m/s]
"""

from __future__ import annotations

import numpy as np

from . import constants as C


def butler_volmer_flux(
    eta: np.ndarray | float,
    j0: float,
    C_local: np.ndarray | float,
    C_ref: float,
    alpha: float = 0.5,
    n: int = 2,
    T: float = C.T_DEFAULT,
) -> np.ndarray | float:
    """Compute Butler-Volmer current density.

    j = j0 * (C_local/C_ref) * [exp(alpha*n*F*eta/(R*T)) - exp(-(1-alpha)*n*F*eta/(R*T))]

    Parameters
    ----------
    eta : array_like
        Overpotential [V].
    j0 : float
        Exchange current density [A/m^2].
    C_local : array_like
        Local concentration [mol/m^3].
    C_ref : float
        Reference concentration [mol/m^3].
    alpha : float
        Charge transfer coefficient.
    n : int
        Number of electrons transferred.
    T : float
        Temperature [K].

    Returns
    -------
    j : array_like
        Current density [A/m^2]. Positive = anodic (oxidation).
    """
    f = n * C.F / (C.R_GAS * T)
    C_ratio = np.clip(np.asarray(C_local) / C_ref, 0, None)
    arg_a = np.clip(alpha * f * np.asarray(eta), -50, 50)
    arg_c = np.clip(-(1 - alpha) * f * np.asarray(eta), -50, 50)
    return j0 * C_ratio * (np.exp(arg_a) - np.exp(arg_c))


def flux_to_consumption_rate(
    j: np.ndarray | float,
    n_electrons: int = 2,
) -> np.ndarray | float:
    """Convert current density to monomer consumption rate.

    R = max(j, 0) / (n * F)  [mol/(m^2*s)]

    Parameters
    ----------
    j : array_like
        Current density [A/m^2].
    n_electrons : int
        Electrons per monomer oxidation.

    Returns
    -------
    R : array_like
        Consumption rate [mol/(m^2*s)].
    """
    return np.maximum(j, 0.) / (n_electrons * C.F)


def flux_to_growth_rate(
    j: np.ndarray | float,
    n_electrons: int = 2,
    M_monomer: float = C.M_PEDOT_REPEAT,
    rho_film: float = C.RHO_PEDOT,
) -> np.ndarray | float:
    """Convert current density to film growth rate.

    dh/dt = max(j, 0) * M / (n * F * rho)  [m/s]

    Parameters
    ----------
    j : array_like
        Current density [A/m^2].
    n_electrons : int
        Electrons per monomer oxidation.
    M_monomer : float
        Deposited repeat-unit molar mass [kg/mol], excluding dopant by default.
    rho_film : float
        Film density [kg/m^3].

    Returns
    -------
    dhdt : array_like
        Growth rate [m/s].
    """
    return np.maximum(j, 0.) * M_monomer / (n_electrons * C.F * rho_film)


def effective_overpotential(
    V_solution: np.ndarray | float,
    V_bpe_float: float,
    E0_formal: float = 0.0,
) -> np.ndarray | float:
    """Compute effective overpotential at the BPE surface.

    eta = V_bpe - V_solution - E0

    Parameters
    ----------
    V_solution : array_like
        Solution potential at electrode surface [V].
    V_bpe_float : float
        Floating potential of BPE [V].
    E0_formal : float
        Formal potential for EDOT oxidation [V].

    Returns
    -------
    eta : array_like
        Overpotential [V].
    """
    return V_bpe_float - np.asarray(V_solution) - E0_formal


def tafel_current(
    eta: np.ndarray | float,
    j0: float,
    alpha: float = 0.5,
    n: int = 2,
    T: float = C.T_DEFAULT,
) -> np.ndarray | float:
    """High-overpotential Tafel approximation (for verification).

    For |eta| >> RT/(nF):
        j_anodic = j0 * exp(alpha*n*F*eta/(R*T))

    Parameters
    ----------
    eta : array_like
        Overpotential [V].
    j0 : float
        Exchange current density [A/m^2].
    alpha : float
    n : int
    T : float

    Returns
    -------
    j : array_like
        Current density [A/m^2].
    """
    f = n * C.F / (C.R_GAS * T)
    arg = np.clip(alpha * f * np.asarray(eta), -50, 50)
    return j0 * np.exp(arg)
