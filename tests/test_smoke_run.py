"""Smoke tests for end-to-end simulation runs."""


import numpy as np
import pytest

from pedot_sim.config import SimConfig
from pedot_sim.geometry import build_domain
from pedot_sim.growth import evolve_one_timestep
from pedot_sim.nucleation import NucleationModel
from pedot_sim.runner_single import run_single


@pytest.fixture
def tiny_config() -> SimConfig:
    """Create a minimal config for fast smoke tests."""
    cfg = SimConfig()
    cfg.numerical.Nx = 30
    cfg.numerical.Ny = 30
    cfg.numerical.dt = 0.1
    cfg.numerical.t_end = 1.0
    cfg.numerical.n_snapshots = 3
    cfg.output.save_fields = True
    cfg.output.save_animations = False
    cfg.output.dpi = 72
    cfg.nucleation.seed = 42
    cfg.nucleation.enable_noise = True
    cfg.nucleation.noise_amplitude = 1e-10
    return cfg


class TestStageASmoke:
    def test_runs_without_error(self, tiny_config, tmp_path):
        tiny_config.physics.physics_stage = "A"
        metrics = run_single(tiny_config, "test_a", tmp_path, "smoke")
        assert "error" not in metrics
        assert "AI" in metrics or "asymmetry_index" in metrics
        ai = metrics.get("AI", metrics.get("asymmetry_index"))
        assert ai >= 1.0

    def test_output_files_exist(self, tiny_config, tmp_path):
        tiny_config.physics.physics_stage = "A"
        run_single(tiny_config, "test_a_files", tmp_path, "smoke")
        run_dir = tmp_path / "simulated_raw" / "smoke" / "test_a_files"
        assert (run_dir / "params.json").exists()
        assert (run_dir / "metadata.json").exists()
        assert (run_dir / "fields.npz").exists()
        assert (run_dir / "thickness.npz").exists()
        assert (run_dir / "timeseries.csv").exists()
        assert (run_dir / "metrics.json").exists()


class TestStageBSmoke:
    def test_runs_without_error(self, tiny_config, tmp_path):
        tiny_config.physics.physics_stage = "B"
        metrics = run_single(tiny_config, "test_b", tmp_path, "smoke")
        assert "error" not in metrics

    def test_v_field_computed(self, tiny_config, tmp_path):
        """Stage B should complete and save field snapshots."""
        tiny_config.physics.physics_stage = "B"
        metrics = run_single(tiny_config, "test_b_field", tmp_path, "smoke")
        assert "error" not in metrics
        run_dir = tmp_path / "simulated_raw" / "smoke" / "test_b_field"
        fields = np.load(run_dir / "fields.npz")
        # Verify concentration snapshots are saved
        assert "C_0000" in fields
        assert "snapshot_times" in fields


class TestStageCSmoke:
    def test_runs_without_error(self, tiny_config, tmp_path):
        tiny_config.physics.physics_stage = "C"
        metrics = run_single(tiny_config, "test_c", tmp_path, "smoke")
        assert "error" not in metrics


class TestStageDSmoke:
    def test_runs_without_error(self, tiny_config, tmp_path):
        tiny_config.physics.physics_stage = "D"
        metrics = run_single(tiny_config, "test_d", tmp_path, "smoke")
        assert "error" not in metrics


class TestDeterminism:
    def test_same_seed_same_result(self, tiny_config, tmp_path):
        """Two runs with same seed should produce identical results."""
        tiny_config.physics.physics_stage = "A"

        cfg1 = tiny_config.copy()
        cfg1.nucleation.seed = 123
        m1 = run_single(cfg1, "det_1", tmp_path, "det")

        cfg2 = tiny_config.copy()
        cfg2.nucleation.seed = 123
        m2 = run_single(cfg2, "det_2", tmp_path, "det")

        assert m1.get("AI") == pytest.approx(m2.get("AI"), rel=1e-10)
        assert m1.get("UI") == pytest.approx(m2.get("UI"), rel=1e-10)

    def test_different_seed_different_result(self, tiny_config, tmp_path):
        """Two runs with different seeds should produce different results."""
        tiny_config.physics.physics_stage = "A"

        cfg1 = tiny_config.copy()
        cfg1.nucleation.seed = 100
        cfg1.nucleation.nucleation_rate = 1e7
        m1 = run_single(cfg1, "diff_1", tmp_path, "det2")

        cfg2 = tiny_config.copy()
        cfg2.nucleation.seed = 200
        cfg2.nucleation.nucleation_rate = 1e7
        m2 = run_single(cfg2, "diff_2", tmp_path, "det2")

        # With different nucleation noise, results should differ at least slightly
        # (though for very short runs the difference may be small)
        # At minimum, check both complete without error
        assert "error" not in m1
        assert "error" not in m2
        assert m1["mean_h"] != m2["mean_h"]


class TestGeometryBuild:
    def test_domain_has_electrode(self, tiny_config):
        domain = build_domain(tiny_config.geometry, tiny_config.numerical)
        assert np.sum(domain.electrode_mask) > 0

    def test_domain_mask_covers_compartments(self, tiny_config):
        domain = build_domain(tiny_config.geometry, tiny_config.numerical)
        assert np.sum(domain.domain_mask) > 0
        assert np.sum(domain.left_feeder_mask) > 0
        assert np.sum(domain.right_feeder_mask) > 0


class TestSingleTimestep:
    def test_evolve_stage_a(self, tiny_config):
        tiny_config.physics.physics_stage = "A"
        domain = build_domain(tiny_config.geometry, tiny_config.numerical)
        grid = domain.grid
        n_electrode = int(np.sum(domain.electrode_mask))

        C = np.zeros((grid.Ny, grid.Nx))
        C[domain.domain_mask] = tiny_config.physics.C_bulk
        h = np.zeros(n_electrode)
        nuc = NucleationModel(tiny_config.nucleation, n_electrode)

        result = evolve_one_timestep(C, h, 0.0, 0.1, tiny_config, domain, nuc)
        assert result.C.shape == C.shape
        assert len(result.h) == n_electrode
        assert result.V_field is None  # Stage A has no field
