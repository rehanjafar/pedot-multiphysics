"""Auditable 3D representations of 2D simulations; no implied 3D physics.

Static PNG/SVG, offline interactive HTML, and OBJ geometry require no OpenGL.
All extrusion dimensions and unit conversions are explicit. A surface with
potential or time as an axis is a scalar/parameter plot, not spatial geometry.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go

from .figure_style import BLUE, ORANGE, scientific_style, save_figure, save_interactive
from .utils_io import save_json


def extruded_network_mesh(owner, height=1.0):
    """Closed exterior of unit-square cells extruded by a stated display height.

    x/y use lattice-cell coordinates; z is illustrative, not measured thickness.
    Shared vertices and omitted internal faces produce a usable OBJ surface.
    """
    owner = np.asarray(owner)
    if owner.ndim != 2 or not np.isin(owner, [0, 1, 2]).all() or not np.any(owner):
        raise ValueError("Network must be a nonempty 2D array with labels 0,1,2")
    if not np.isfinite(height) or height <= 0:
        raise ValueError("Illustrative extrusion height must be positive and finite")
    vertices, triangles, labels, lookup = [], [], [], {}

    def quad(points, label):
        indices = []
        for point in points:
            if point not in lookup:
                lookup[point] = len(vertices)
                vertices.append(point)
            indices.append(lookup[point])
        a, b, c, d = indices
        triangles.extend([(a, b, c), (a, c, d)])
        labels.extend([label, label])

    ny, nx = owner.shape
    for y, x in np.argwhere(owner > 0):
        label = int(owner[y, x])
        a, b = x - 0.5, x + 0.5
        c, d = y - 0.5, y + 0.5
        quad([(a, c, 0.0), (a, d, 0.0), (b, d, 0.0), (b, c, 0.0)], label)
        quad([(a, c, height), (b, c, height), (b, d, height), (a, d, height)], label)
        if x == 0 or owner[y, x - 1] == 0:
            quad([(a, c, 0.0), (a, c, height), (a, d, height), (a, d, 0.0)], label)
        if x == nx - 1 or owner[y, x + 1] == 0:
            quad([(b, c, 0.0), (b, d, 0.0), (b, d, height), (b, c, height)], label)
        if y == 0 or owner[y - 1, x] == 0:
            quad([(a, c, 0.0), (b, c, 0.0), (b, c, height), (a, c, height)], label)
        if y == ny - 1 or owner[y + 1, x] == 0:
            quad([(a, d, 0.0), (a, d, height), (b, d, height), (b, d, 0.0)], label)
    return np.array(vertices, float), np.array(triangles, int), np.array(labels, int)


def export_obj(path, vertices, triangles, description):
    """OBJ is a geometry exchange file; its explanatory header must be retained."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# " + description.replace("\n", " ") + "\n")
        for x, y, z in vertices:
            f.write(f"v {x:.17g} {y:.17g} {z:.17g}\n")
        for a, b, c in triangles:
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


def network_3d(owner, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    vertices, faces, labels = extruded_network_mesh(owner)
    export_obj(
        directory / "network_illustrative.obj",
        vertices,
        faces,
        "Illustrative one-cell extrusion of a 2D model. x/y=lattice cells; z=assumed display depth. Not measured 3D PEDOT.",
    )
    colors = np.where(labels == 1, ORANGE, BLUE)
    with scientific_style():
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        ax.add_collection3d(
            Poly3DCollection(vertices[faces], facecolors=colors, edgecolors="none", alpha=1.0)
        )
        ax.set(
            xlim=(-0.5, owner.shape[1] - 0.5),
            ylim=(-0.5, owner.shape[0] - 0.5),
            zlim=(0, 1.5),
            zticks=[0, 1],
            xlabel="x [lattice cells]",
            ylabel="y [lattice cells]",
            zlabel="Assumed display depth",
            title="2D conducting network · illustrative extrusion",
        )
        ax.set_box_aspect((owner.shape[1], owner.shape[0], 4))
        ax.view_init(35, -55)
        entry = save_figure(
            fig,
            directory,
            "network_3d",
            "How are the two conducting networks arranged?",
            "Each occupied 2D cell is extruded by one display cell. Orange is driven; blue is grounded. No physical thickness, volume or 3D growth is inferred.",
            "illustrative 3D view",
        )
    traces = []
    for label, color, name in [(1, ORANGE, "Driven network"), (2, BLUE, "Grounded network")]:
        selected = faces[labels == label]
        traces.append(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=selected[:, 0],
                j=selected[:, 1],
                k=selected[:, 2],
                color=color,
                name=name,
                showlegend=True,
                flatshading=True,
                hovertemplate="x=%{x} cells<br>y=%{y} cells<br>Illustrative z=%{z}<extra>"
                + name
                + "</extra>",
            )
        )
    entry["interactive"] = save_interactive(
        go.Figure(traces),
        directory,
        "network_3d",
        "Conducting network: illustrative 3D extrusion",
        "The simulated geometry is 2D; depth is a display assumption.",
        {
            "xaxis_title": "x [cells]",
            "yaxis_title": "y [cells]",
            "zaxis_title": "Assumed depth [display cells]",
            "aspectmode": "data",
        },
    )
    save_json(
        {
            "geometry": "extrusion of 2D occupancy",
            "height": 1.0,
            "height_basis": "illustrative lattice-cell depth",
            "physical_volume_valid": False,
            "vertices": len(vertices),
            "triangles": len(faces),
        },
        directory / "network_3d_metadata.json",
    )
    return entry


def scalar_surface(x, y, z, directory, name, title, zlabel, caption, origin="model field"):
    """Full-resolution scalar surface; no spatial subsampling or smoothing."""
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    if z.shape != (len(y), len(x)) or not np.isfinite(z).any():
        raise ValueError("Scalar surface must match coordinates and contain finite values")
    X, Y = np.meshgrid(x, y)
    with scientific_style():
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        surface = ax.plot_surface(
            X, Y, z, cmap="viridis", rcount=len(y), ccount=len(x), linewidth=0, antialiased=False
        )
        ax.set(xlabel="x", ylabel="y", zlabel=zlabel, title=title)
        ax.view_init(30, -60)
        fig.colorbar(surface, ax=ax, shrink=0.55, pad=0.12, label=zlabel)
        entry = save_figure(fig, directory, name, title, caption, origin)
    return entry


def plot_thickness_3d(h, electrode_x, save_path):
    """Backwards-compatible static profile extrusion with an explicit width."""
    h, x = np.asarray(h, float), np.asarray(electrode_x, float)
    if (
        h.shape != x.shape
        or len(x) < 2
        or not np.isfinite(h).all()
        or np.any(h < 0)
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("Thickness and increasing x coordinates must be finite and matching")
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with scientific_style():
        fig = plt.figure(figsize=(8, 5), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        X, Y = np.meshgrid(x * 1e3, [0.0, 0.1])
        Z = np.tile(h * 1e6, (2, 1))
        ax.plot_surface(X, Y, Z, cmap="viridis", rcount=2, ccount=len(x))
        ax.set(
            xlabel="x [mm]",
            ylabel="Illustrative width [mm]",
            zlabel="Model thickness [µm]",
            title="1D thickness extruded over an assumed 0.1 mm width",
        )
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def thickness_profile_3d(h, x, directory):
    """Single-profile fallback; display width is explicitly arbitrary."""
    x, h = np.asarray(x), np.asarray(h)
    with scientific_style():
        fig = plt.figure(figsize=(9, 6), layout="constrained")
        ax = fig.add_subplot(projection="3d")
        X, Y = np.meshgrid(x * 1e3, [0.0, 0.1])
        Z = np.tile(h * 1e6, (2, 1))
        ax.plot_surface(X, Y, Z, cmap="viridis", rcount=2, ccount=len(x))
        ax.set(
            xlabel="Position [mm]",
            ylabel="Assumed width [mm]",
            zlabel="Model thickness [µm]",
            title="Final thickness · assumed 0.1 mm display width",
        )
        entry = save_figure(
            fig,
            directory,
            "thickness_profile_3d",
            "What is the final thickness profile?",
            "The 1D profile is extruded across an arbitrary 0.1 mm width. Axis units differ and vertical scale is exaggerated. No physical volume is inferred.",
            "illustrative profile extrusion",
        )
    entry["interactive"] = save_interactive(
        go.Figure(go.Surface(x=x * 1e3, y=[0.0, 0.1], z=Z, colorscale="Viridis")),
        directory,
        "thickness_profile_3d",
        "Final model thickness",
        "Arbitrary display width; vertical scale is exaggerated.",
        {
            "xaxis_title": "Position [mm]",
            "yaxis_title": "Assumed width [mm]",
            "zaxis_title": "Thickness [µm]",
        },
    )
    return entry


def animate_growth(h_snapshots, electrode_x, save_path, fps=20, snapshot_times=None):
    """GIF of recorded profiles; playback is compressed, timestamps are real.

    Unequal recorded intervals receive proportional frame durations (rounded
    to GIF's 10 ms resolution). No temporal interpolation of profiles.
    """
    from io import BytesIO
    from PIL import Image

    h, x = np.asarray(h_snapshots), np.asarray(electrode_x)
    times = np.asarray(snapshot_times) if snapshot_times is not None else None
    if (
        times is None
        or h.shape != (len(times), len(x))
        or len(times) < 2
        or not np.all(np.diff(times) > 0)
    ):
        raise ValueError(
            "Animation requires at least two profiles and their increasing real timestamps"
        )
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Animation frame rate must be positive")
    frames = []
    with scientific_style():
        for time, profile in zip(times, h):
            fig = plt.figure(figsize=(7, 5), layout="constrained")
            ax = fig.add_subplot(projection="3d")
            X, Y = np.meshgrid(x * 1e3, [0.0, 0.1])
            ax.plot_surface(
                X,
                Y,
                np.tile(profile * 1e6, (2, 1)),
                cmap="viridis",
                vmin=0,
                vmax=max(h.max() * 1e6, 1e-9),
                rcount=2,
                ccount=len(x),
            )
            ax.set(
                xlabel="Position [mm]",
                ylabel="Assumed width [mm]",
                zlabel="Thickness [µm]",
                zlim=(0, max(h.max() * 1e6, 1e-9)),
                title=f"Recorded t = {time:.6g} s · illustrative width",
            )
            with BytesIO() as buffer:
                fig.savefig(buffer, format="png", dpi=100)
                buffer.seek(0)
                frames.append(Image.open(buffer).convert("RGB"))
            plt.close(fig)
    delta = np.diff(times)
    durations = np.r_[delta, delta[-1]]
    durations = (
        np.maximum(10, np.round(durations / np.mean(durations) * 100 / fps) * 10)
        .astype(int)
        .tolist()
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    return Path(save_path)


def generate_3d_outputs(
    run_dir, h_snapshots, h_final, electrode_x, grid=None, fps=20, snapshot_times=None
):
    """Produce static and animated views without optional OpenGL dependencies."""
    directory = Path(run_dir) / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    plot_thickness_3d(h_final, electrode_x, directory / "thickness_3d.png")
    if snapshot_times is not None and len(snapshot_times) >= 2:
        return animate_growth(
            h_snapshots, electrode_x, directory / "growth_3d.gif", fps, snapshot_times
        )
    return None
