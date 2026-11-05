"""
Structured 2D grid utilities for the simulation domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid2D:
    """Uniform 2D structured grid.

    x runs along the horizontal (left-to-right) direction.
    y runs along the vertical direction.

    Attributes
    ----------
    x : np.ndarray
        1D array of x coordinates [m], shape (Nx,).
    y : np.ndarray
        1D array of y coordinates [m], shape (Ny,).
    X : np.ndarray
        2D meshgrid of x, shape (Ny, Nx).
    Y : np.ndarray
        2D meshgrid of y, shape (Ny, Nx).
    dx : float
        Grid spacing in x [m].
    dy : float
        Grid spacing in y [m].
    Nx : int
        Number of grid points in x.
    Ny : int
        Number of grid points in y.
    """
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float
    Nx: int
    Ny: int


def create_grid(Lx: float, Ly: float, Nx: int, Ny: int) -> Grid2D:
    """Create a uniform 2D grid.

    Parameters
    ----------
    Lx : float
        Domain length in x [m].
    Ly : float
        Domain length in y [m].
    Nx : int
        Number of points in x.
    Ny : int
        Number of points in y.

    Returns
    -------
    Grid2D
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    dx = Lx / max(Nx - 1, 1)
    dy = Ly / max(Ny - 1, 1)
    return Grid2D(x=x, y=y, X=X, Y=Y, dx=dx, dy=dy, Nx=Nx, Ny=Ny)
