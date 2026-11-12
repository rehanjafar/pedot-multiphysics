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


def cell_widths(grid):
    """Numerical control-volume widths; uniform legacy grids retain their measure."""
    return (getattr(grid, "widths_x", np.full(grid.Nx, grid.dx)),
            getattr(grid, "widths_y", np.full(grid.Ny, grid.dy)))


def cell_areas(grid):
    wx, wy = cell_widths(grid)
    return wy[:, None]*wx[None, :]


def electrode_weights(grid, mask):
    wx, _ = cell_widths(grid)
    return np.broadcast_to(wx, mask.shape)[mask]


def create_refined_grid(Lx, Ly, Nx, Ny, anchors_x, anchors_y, strength):
    """Cell-centred, boundary-fitted grid clustered at reaction interfaces.

    Each interval between anchors uses symmetric sinh grading. Exact anchor edges
    preserve compartment, bridge and electrode dimensions under refinement.
    Nx/Ny are target resolution budgets, distributed equally among anchored
    intervals to retain resolution at small electrodes and narrow bridges.
    """
    def edges(length, count, anchors):
        anchors = np.unique(np.r_[0., anchors, length])
        if anchors[0] < 0 or anchors[-1] > length:
            raise ValueError("Grid anchor outside domain")
        chunks = []
        for low, high in zip(anchors[:-1], anchors[1:]):
            n = max(4, int(np.ceil(count/(len(anchors)-1))))
            # Small cells at both ends of an interval, coarser in its centre.
            u = np.linspace(-1., 1., n+1)
            if strength:
                v = np.sign(u)*(1-np.sinh(strength*(1-np.abs(u)))/np.sinh(strength))
            else:
                v = u
            chunks.append(low+(high-low)*(v+1)/2)
        return np.unique(np.concatenate(chunks))
    xe = edges(Lx, Nx, anchors_x)
    ye = edges(Ly, Ny, anchors_y)
    x, y = (xe[:-1]+xe[1:])/2, (ye[:-1]+ye[1:])/2
    X, Y = np.meshgrid(x, y, indexing="xy")
    grid = Grid2D(x, y, X, Y, float(np.min(np.diff(xe))), float(np.min(np.diff(ye))), len(x), len(y))
    grid.widths_x, grid.widths_y = np.diff(xe), np.diff(ye)
    return grid
