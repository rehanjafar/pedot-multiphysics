"""
Two-compartment salt-bridge domain geometry builder.

Domain layout (top view):
  +-----------+     +---+     +-----------+
  |           |     |   |     |           |
  |   LEFT    |     | B |     |   RIGHT   |
  |  COMPART- |=====| R |=====|  COMPART- |
  |   MENT    |     | I |     |   MENT    |
  |           |     | D |     |           |
  |  (feeder  |     | G |     |  (feeder  |
  |   anode)  |     | E |     |  cathode) |
  +-----------+     +---+     +-----------+

  x --->  (horizontal: left to right across the full domain)
  y |     (vertical)

The bipolar electrode (BPE) sits as a horizontal strip inside the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import GeometryConfig, NumericalConfig
from .utils_grid import Grid2D, create_grid


@dataclass
class DomainInfo:
    """Complete domain description with grid and masks.

    Attributes
    ----------
    grid : Grid2D
        The structured 2D grid.
    domain_mask : np.ndarray
        Bool array (Ny, Nx). True = active cell, False = outside domain.
    region_masks : dict
        Dict of bool arrays for named regions.
    electrode_mask : np.ndarray
        Bool array marking the BPE boundary cells.
    left_feeder_mask : np.ndarray
        Bool array marking left feeder electrode boundary.
    right_feeder_mask : np.ndarray
        Bool array marking right feeder electrode boundary.
    wall_mask : np.ndarray
        Bool array marking insulating wall cells.
    Lx : float
        Total domain length in x [m].
    Ly : float
        Total domain length in y [m].
    """
    grid: Grid2D
    domain_mask: np.ndarray
    region_masks: Dict[str, np.ndarray]
    electrode_mask: np.ndarray
    left_feeder_mask: np.ndarray
    right_feeder_mask: np.ndarray
    wall_mask: np.ndarray
    Lx: float
    Ly: float


def build_domain(geo: GeometryConfig, num: NumericalConfig) -> DomainInfo:
    """Build the two-compartment salt-bridge domain.

    The total domain spans:
      x: [0, 2*compartment_width + bridge_length]
      y: [0, compartment_height]

    Regions:
      Left compartment : x in [0, compartment_width]
      Bridge           : x in [compartment_width, compartment_width + bridge_length]
      Right compartment: x in [compartment_width + bridge_length, 2*cw + bl]

    The bridge occupies only a central vertical band:
      y in [y_center - bridge_width/2, y_center + bridge_width/2]
    where y_center = compartment_height / 2.

    The BPE is a horizontal strip at the center of the bridge:
      x in [x_bridge_center - electrode_length/2, x_bridge_center + electrode_length/2]
      y at the bridge centerline (single row of cells).
    """
    cw = geo.compartment_width
    ch = geo.compartment_height
    bl = geo.bridge_length
    bw = geo.bridge_width
    el = geo.electrode_length

    Lx = 2 * cw + bl
    Ly = ch

    grid = create_grid(Lx, Ly, num.Nx, num.Ny)

    y_center = Ly / 2.0
    bridge_y_lo = y_center - bw / 2.0
    bridge_y_hi = y_center + bw / 2.0
    bridge_x_lo = cw
    bridge_x_hi = cw + bl

    # Region masks (Ny, Nx)
    left_mask = grid.X <= cw
    right_mask = grid.X >= cw + bl
    bridge_x = (grid.X > cw) & (grid.X < cw + bl)
    bridge_y = (grid.Y >= bridge_y_lo) & (grid.Y <= bridge_y_hi)
    bridge_mask = bridge_x & bridge_y

    # Domain = left + right compartments + bridge channel
    domain_mask = left_mask | right_mask | bridge_mask

    from scipy.ndimage import label
    _, components = label(domain_mask)
    if components != 1:
        raise ValueError("Grid does not resolve a connected salt bridge; increase Ny")

    # Electrode (BPE) at bridge center
    x_bpe_center = cw + bl / 2.0
    bpe_x = (grid.X >= x_bpe_center - el / 2.0) & (grid.X <= x_bpe_center + el / 2.0)
    # Find the y-index closest to centerline
    j_center = np.argmin(np.abs(grid.y - y_center))
    electrode_mask = np.zeros_like(domain_mask)
    bpe_x_row = bpe_x[j_center, :]
    electrode_mask[j_center, :] = bpe_x_row
    electrode_mask = electrode_mask & domain_mask

    # Feeder electrodes: leftmost and rightmost columns inside domain
    left_feeder = np.zeros_like(domain_mask)
    left_feeder[:, 0] = domain_mask[:, 0]

    right_feeder = np.zeros_like(domain_mask)
    right_feeder[:, -1] = domain_mask[:, -1]

    # Wall mask: domain boundary cells that are not feeders and not open
    # Identified as domain cells adjacent to non-domain cells
    wall_mask = np.zeros_like(domain_mask)
    padded = np.pad(domain_mask.astype(int), 1, mode="constant", constant_values=0)
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        neighbor = padded[1 + di: padded.shape[0] - 1 + di,
                          1 + dj: padded.shape[1] - 1 + dj]
        wall_mask |= domain_mask & (neighbor == 0)
    # Remove feeder locations from wall
    wall_mask = wall_mask & (~left_feeder) & (~right_feeder)

    region_masks = {
        "left_compartment": left_mask & domain_mask,
        "right_compartment": right_mask & domain_mask,
        "bridge": bridge_mask,
    }

    return DomainInfo(
        grid=grid,
        domain_mask=domain_mask,
        region_masks=region_masks,
        electrode_mask=electrode_mask,
        left_feeder_mask=left_feeder,
        right_feeder_mask=right_feeder,
        wall_mask=wall_mask,
        Lx=Lx,
        Ly=Ly,
    )


def get_electrode_x(domain: DomainInfo) -> np.ndarray:
    """Return x-coordinates along the BPE surface.

    Returns 1D array of x values where electrode_mask is True (at its y-row).
    """
    rows, cols = np.where(domain.electrode_mask)
    if len(cols) == 0:
        return np.array([])
    return domain.grid.x[np.sort(np.unique(cols))]
