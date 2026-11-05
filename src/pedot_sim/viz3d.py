"""
3D visualization module using PyVista.

Generates:
- 3D surface plots of film thickness
- Growth animation (MP4 + GIF fallback)
- Rotating camera animation

Gracefully degrades if PyVista or ffmpeg is not available.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from .utils_grid import Grid2D
from .utils_io import ensure_dir

logger = logging.getLogger(__name__)

# Try to import PyVista
try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    logger.info("PyVista not available. 3D visualizations will be skipped.")

# Try to import imageio for GIF
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def _make_surface_grid(h: np.ndarray, electrode_x: np.ndarray) -> "pv.StructuredGrid":
    """Create PyVista structured grid from thickness profile.

    Parameters
    ----------
    h : np.ndarray, shape (n_electrode,)
        Film thickness [m].
    electrode_x : np.ndarray, shape (n_electrode,)
        x-coordinates [m].

    Returns
    -------
    pv.StructuredGrid
    """
    n = len(electrode_x)
    ny = 2  # Extrude slightly in y for 3D effect
    x = np.tile(electrode_x * 1e3, (ny, 1))  # mm
    y_vals = np.array([0, 0.1])  # mm, thin strip
    y = np.repeat(y_vals[:, np.newaxis], n, axis=1)
    z = np.tile(h * 1e6, (ny, 1))  # µm

    grid = pv.StructuredGrid(x, y, z)
    grid.point_data["thickness_um"] = z.ravel()
    return grid


def plot_thickness_3d(
    h: np.ndarray,
    electrode_x: np.ndarray,
    save_path: Path,
) -> None:
    """3D surface plot of final film thickness."""
    if not HAS_PYVISTA:
        logger.warning("PyVista not available, skipping 3D thickness plot")
        return

    save_path = Path(save_path)
    grid = _make_surface_grid(h, electrode_x)

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.add_mesh(grid, scalars="thickness_um", cmap="viridis",
                     show_edges=False, lighting=True)
    plotter.add_axes()
    plotter.add_text("PEDOT Film Thickness", font_size=14)
    plotter.camera_position = "xz"
    plotter.camera.elevation = 30
    plotter.screenshot(str(save_path))
    plotter.close()
    logger.info("3D thickness plot saved to %s", save_path)


def animate_growth(
    h_snapshots: np.ndarray,
    electrode_x: np.ndarray,
    save_dir: Path,
    fps: int = 20,
) -> None:
    """Create growth animation.

    Parameters
    ----------
    h_snapshots : np.ndarray, shape (n_times, n_electrode)
    electrode_x : np.ndarray, shape (n_electrode,)
    save_dir : Path
    fps : int
    """
    if not HAS_PYVISTA:
        logger.warning("PyVista not available, skipping growth animation")
        # Fallback: matplotlib animation as GIF
        _animate_growth_matplotlib(h_snapshots, electrode_x, save_dir, fps)
        return

    save_dir = Path(save_dir)
    n_frames = h_snapshots.shape[0]
    frames = []

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    h_max = np.max(h_snapshots) * 1e6 if np.max(h_snapshots) > 0 else 1.0

    for i in range(n_frames):
        plotter.clear()
        grid = _make_surface_grid(h_snapshots[i], electrode_x)
        plotter.add_mesh(grid, scalars="thickness_um", cmap="viridis",
                         clim=[0, h_max], show_edges=False)
        plotter.add_text(f"Frame {i + 1}/{n_frames}", font_size=12)
        plotter.camera_position = "xz"
        plotter.camera.elevation = 30
        img = plotter.screenshot(return_img=True)
        frames.append(img)

    plotter.close()

    # Save as GIF
    if HAS_IMAGEIO and frames:
        gif_path = save_dir / "growth_animation.gif"
        imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
        logger.info("Growth animation GIF saved to %s", gif_path)

    # Try MP4
    try:
        if HAS_IMAGEIO and frames:
            mp4_path = save_dir / "growth_animation.mp4"
            writer = imageio.get_writer(str(mp4_path), fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            logger.info("Growth animation MP4 saved to %s", mp4_path)
    except Exception as e:
        logger.warning("MP4 export failed (ffmpeg may not be available): %s", e)


def _animate_growth_matplotlib(
    h_snapshots: np.ndarray,
    electrode_x: np.ndarray,
    save_dir: Path,
    fps: int = 10,
) -> None:
    """Fallback matplotlib-based growth animation as GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    n_frames = h_snapshots.shape[0]
    h_max = np.max(h_snapshots) * 1e6 if np.max(h_snapshots) > 0 else 1.0

    frames = []
    for i in range(n_frames):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.fill_between(electrode_x * 1e3, 0, h_snapshots[i] * 1e6,
                         alpha=0.7, color="steelblue")
        ax.plot(electrode_x * 1e3, h_snapshots[i] * 1e6, "b-", linewidth=2)
        ax.set_xlim(electrode_x[0] * 1e3, electrode_x[-1] * 1e3)
        ax.set_ylim(0, h_max * 1.1)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("Thickness [µm]")
        ax.set_title(f"PEDOT Growth (frame {i + 1}/{n_frames})")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        plt.close(fig)

    if HAS_IMAGEIO and frames:
        gif_path = save_dir / "growth_animation.gif"
        imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
        logger.info("Matplotlib growth animation saved to %s", gif_path)


def animate_rotating(
    h: np.ndarray,
    electrode_x: np.ndarray,
    save_dir: Path,
    fps: int = 20,
    n_frames: int = 60,
) -> None:
    """Rotating camera animation of final morphology."""
    if not HAS_PYVISTA:
        logger.warning("PyVista not available, skipping rotating animation")
        return

    save_dir = Path(save_dir)
    grid = _make_surface_grid(h, electrode_x)
    frames = []

    plotter = pv.Plotter(off_screen=True, window_size=(1200, 800))
    plotter.add_mesh(grid, scalars="thickness_um", cmap="viridis", show_edges=False)
    plotter.add_text("PEDOT Film - Rotating View", font_size=12)

    for i in range(n_frames):
        azimuth = 360 * i / n_frames
        plotter.camera_position = "xz"
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = 30
        img = plotter.screenshot(return_img=True)
        frames.append(img)

    plotter.close()

    if HAS_IMAGEIO and frames:
        gif_path = save_dir / "rotating_morphology.gif"
        imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
        logger.info("Rotating animation saved to %s", gif_path)


def generate_3d_outputs(
    run_dir: Path,
    h_snapshots: np.ndarray,
    h_final: np.ndarray,
    electrode_x: np.ndarray,
    grid: Grid2D,
    fps: int = 20,
) -> None:
    """Generate all 3D outputs for a run."""
    if len(electrode_x) == 0 or len(h_final) == 0:
        logger.warning("No electrode data for 3D visualization")
        return

    fig_dir = ensure_dir(Path(run_dir) / "figures")
    anim_dir = ensure_dir(Path(run_dir) / "animations")

    plot_thickness_3d(h_final, electrode_x, fig_dir / "thickness_3d.png")
    animate_growth(h_snapshots, electrode_x, anim_dir, fps)
    animate_rotating(h_final, electrode_x, anim_dir, fps)
