"""Numerical identity, provenance and actual export checks for the evidence views."""

from collections import Counter
from html.parser import HTMLParser
import hashlib
import json

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from pedot_sim.literature import load_reference_tables, curate_source, analyze_literature
from pedot_sim.viz3d import extruded_network_mesh, export_obj, animate_growth
from pedot_sim.metrics import mean_growth_rate


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if (
                key in ("href", "src")
                and value
                and not value.startswith(("http:", "https:", "#", "data:"))
            ):
                self.links.append(value)


def assert_gallery(root):
    parser = Links()
    parser.feed((root / "index.html").read_text())
    assert parser.links
    for link in parser.links:
        assert (root / link).is_file(), link
    manifest = json.loads((root / "figure_manifest.json").read_text())
    for figure in manifest["figures"]:
        with Image.open(root / figure["png"]) as im:
            assert im.width >= 1000 and im.height >= 500
            assert np.asarray(im.convert("RGB")).std() > 10
        assert "<svg" in (root / figure["svg"]).read_text()
        if "interactive" in figure:
            parser = Links()
            parser.feed((root / figure["interactive"]).read_text())
            assert "plotly.min.js" in parser.links
            for link in parser.links:
                assert (root / link).is_file()
    return manifest


@pytest.mark.parametrize(
    "owner,height",
    [
        (np.array([[1]]), 1.0),
        (np.array([[1, 1]]), 2.5),
        (np.array([[1, 0], [1, 1]]), 0.3),
        (np.array([[1, 2], [1, 2]]), 1.0),
    ],
)
def test_mesh_closed_oriented_volume_and_shared_faces(owner, height, tmp_path):
    vertices, faces, labels = extruded_network_mesh(owner, height)
    edges = Counter(
        tuple(sorted((a, b))) for face in faces for a, b in zip(face, np.roll(face, -1))
    )
    assert set(edges.values()) == {2}
    triangles = vertices[faces]
    volume = (
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6
    )
    assert volume == pytest.approx(np.count_nonzero(owner) * height)
    assert set(labels) == set(owner[owner > 0])
    target = tmp_path / "mesh.obj"
    export_obj(target, vertices, faces, "Illustrative, not measured 3D")
    exported = np.array(
        [
            list(map(int, line.split()[1:]))
            for line in target.read_text().splitlines()
            if line.startswith("f ")
        ]
    )
    np.testing.assert_array_equal(exported, faces + 1)


@pytest.mark.parametrize(
    "owner,height",
    [
        (np.zeros((2, 2)), 1),
        (np.array([[3]]), 1),
        (np.array([1, 2]), 1),
        (np.array([[1]]), 0),
        (np.array([[1]]), float("nan")),
    ],
)
def test_reject_invalid_illustrative_mesh(owner, height):
    with pytest.raises(ValueError):
        extruded_network_mesh(owner, height)


def test_graded_surface_average_growth_rate():
    h = np.array([[0.0, 0.0], [2.0, 10.0]])
    assert mean_growth_rate(h, np.array([0.0, 2.0]), np.array([3.0, 1.0])) == 2.0


def test_hash_failure_precedes_extraction_or_output_creation(tmp_path):
    source = tmp_path / "bad.csv"
    source.write_text("unreviewed data")
    with pytest.raises(ValueError, match="SHA-256"):
        curate_source(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_reference_values_units_categories_and_anomaly():
    tables, meta = load_reference_tables()
    assert {key: len(df) for key, df in tables.items()} == dict(
        amplitude_morphology=7,
        frequency_morphology=12,
        offset_morphology=10,
        growth_traces=672,
        impedance=183,
        cyclic_voltammetry=1254,
    )
    row = tables["amplitude_morphology"].iloc[-1]
    assert row.amplitude_V == 6.5 and row.completion_time_s == 206
    assert row.projected_area_m2 == pytest.approx(0.038436735e-6, rel=1e-14)
    assert row.source_row_1based == 801
    f = tables["frequency_morphology"]
    flagged = f[f.quality_status != "as_reported"]
    assert len(flagged) == 1 and flagged.iloc[0].frequency_Hz == 10
    assert flagged.iloc[0].projected_area_m2 == 0
    assert flagged.iloc[0].image_fractal_number == 1.24499
    assert "circuit modeling and fits" in meta["changes"][0]
    assert meta["license"] == "CC BY 4.0"


@pytest.mark.parametrize("table", ["amplitude_morphology", "frequency_morphology"])
def test_published_branching_sd_is_six_spatial_samples(table):
    tables, _ = load_reference_tables()
    df = tables[table]
    vals = df[[f"branching_position_{n}" for n in range(1, 7)]].to_numpy()
    np.testing.assert_allclose(vals.mean(axis=1), df.branching_mean, atol=1e-9, rtol=0)
    np.testing.assert_allclose(vals.std(axis=1, ddof=1), df.branching_spatial_sd, atol=1e-9, rtol=0)


def test_cv_preserves_forward_and_reverse_acquisition_order():
    tables, _ = load_reference_tables()
    for _, df in tables["cyclic_voltammetry"].groupby("chemistry"):
        assert (df.cycle == 1).all()
        assert (np.diff(df.source_row_1based) > 0).all()
        delta = np.diff(df.potential_V_vs_AgAgCl)
        assert (delta > 0).any() and (delta < 0).any()
    first = tables["impedance"].iloc[0]
    assert (
        first.sample_id,
        first.frequency_Hz,
        first.impedance_modulus_ohm,
        first.impedance_phase_deg,
    ) == (1, 1e6, 65125.1, -15.69219)


@pytest.fixture(scope="module")
def literature_atlas(tmp_path_factory):
    root = tmp_path_factory.mktemp("literature")
    analyze_literature(root)
    return root


def test_published_atlas_exports_are_complete_and_preserve_checksums(literature_atlas):
    root = literature_atlas
    manifest = assert_gallery(root)
    assert len(manifest["figures"]) == 7
    meta = json.loads((root / "source_metadata.json").read_text())
    for table, info in meta["sections"].items():
        assert hashlib.sha256((root / (table + ".csv")).read_bytes()).hexdigest() == info["sha256"]
    data = json.loads((root / "growth_parameter_3d.plotly.json").read_text())["data"]
    frame = pd.read_csv(root / "growth_traces.csv")
    for trace, (_, group) in zip(data, frame.groupby("amplitude_V")):
        np.testing.assert_array_equal(trace["x"], group.time_s)
        np.testing.assert_allclose(trace["z"], group.projected_area_m2 * 1e6, rtol=1e-13)
        assert trace["type"] == "scatter3d"  # No surface between measured conditions.
    cart = pd.read_csv(root / "impedance_cartesian.csv")
    np.testing.assert_allclose(
        np.hypot(cart.real_impedance_ohm, cart.negative_imaginary_impedance_ohm),
        cart.impedance_modulus_ohm,
        rtol=1e-14,
    )


def test_archived_potential_corresponds_to_pre_attachment_geometry():
    from pedot_sim.dendrites import DendriteConfig, simulate_dendrites, solve_network_potential

    c = DendriteConfig(
        nx=25,
        ny=17,
        wire_length=4,
        wire_width=3,
        n_particles=180,
        steps=20,
        attachment_probability=1.0,
    )
    result = simulate_dendrites(c)
    owner = result["potential_owner"]
    v = result["potential_V"]
    drive = float(v[owner == 1][0])
    expected, residual = solve_network_potential(owner, drive)
    np.testing.assert_allclose(v, expected, atol=2e-13)
    assert residual < 1e-10
    assert result["potential_time"] < result["metrics"]["elapsed_model_steps"]
    assert np.count_nonzero(owner) <= np.count_nonzero(result["owner"])


def test_strip_actual_gallery_every_step_budget_and_no_snapshot_fallback(tmp_path):
    from pedot_sim.config import SimConfig
    from pedot_sim.runner_single import run_single

    c = SimConfig()
    c.numerical.Nx = c.numerical.Ny = 30
    c.numerical.t_end = 0.03
    c.numerical.n_snapshots = 0
    c.output.save_animations = False
    result = run_single(c, "small", tmp_path, "visual")
    assert "error" not in result, result
    root = tmp_path / "simulated_raw/visual/small"
    meta = json.loads((root / "metadata.json").read_text())
    budget = pd.read_csv(root / "integration_diagnostics.csv")
    assert len(budget) == meta["n_timesteps"]
    assert budget.time_end_s.iloc[-1] == 0.03
    assert budget.consumed_mol_per_m.iloc[-1] == pytest.approx(
        result["consumed_mol_per_m"], rel=1e-12
    )
    assert budget.balance_error_mol_per_m.abs().max() < 1e-12
    manifest = assert_gallery(root / "views")
    assert "thickness_profile_3d" in [f["name"] for f in manifest["figures"]]


def test_dendrite_actual_gallery_and_saved_fields(tmp_path):
    from pedot_sim.dendrites import DendriteConfig, run_dendrite

    c = DendriteConfig(nx=25, ny=17, wire_length=4, n_particles=80, steps=3)
    run_dendrite(c, "small", tmp_path, "visual")
    root = tmp_path / "dendrites/visual/small"
    manifest = assert_gallery(root / "views")
    assert len(manifest["figures"]) == 6
    with np.load(root / "morphology.npz") as state:
        assert state["potential_V"].shape == state["owner"].shape
    assert (
        json.loads((root / "views/network_3d_metadata.json").read_text())["physical_volume_valid"]
        is False
    )


def test_growth_animation_uses_recorded_profiles_and_exports(tmp_path):
    x = np.linspace(0, 0.001, 3)
    h = np.array([[0.0, 0.0, 0.0], [1e-6, 2e-6, 1e-6], [2e-6, 3e-6, 2e-6]])
    path = animate_growth(h, x, tmp_path / "growth.gif", fps=4, snapshot_times=[0.0, 0.1, 0.4])
    with Image.open(path) as im:
        assert im.n_frames == 3
        durations = []
        for n in range(3):
            im.seek(n)
            durations.append(im.info["duration"])
    assert durations[1] > durations[0] * 2.5
    with pytest.raises(ValueError):
        animate_growth(h, x, path, snapshot_times=[0.0, 0.0, 1.0])


def test_manufacturing_cumulative_energy_equals_exact_result(tmp_path):
    from pedot_sim.manufacturing import analyze_manufacturing

    frame = pd.DataFrame(
        {"time_s": [0.0, 1.0, 3.0], "voltage_V": [-2.0, 3.0, -1.0], "current_A": [1.0, -2.0, 3.0]}
    )
    trace = tmp_path / "trace.csv"
    frame.to_csv(trace, index=False)
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "source": "synthetic test",
                "units": {"time_s": "s", "voltage_V": "V", "current_A": "A"},
            }
        )
    )
    out = tmp_path / "analysis"
    result = analyze_manufacturing(trace, meta, out)
    cumulative = pd.read_csv(out / "cumulative_electrical_energy.csv").iloc[-1]
    for key in ("net", "imported", "returned"):
        assert cumulative[key + "_energy_J"] == pytest.approx(
            result["metrics"][key + "_terminal_energy_J"], abs=1e-14
        )
    assert_gallery(out / "views")
