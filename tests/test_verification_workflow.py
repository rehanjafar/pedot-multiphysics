"""Verification must retain failed-study evidence and refine on the finest mesh."""
import json
import pytest
from pedot_sim.config import SimConfig
from pedot_sim import verification


def test_failed_runs_write_failed_verification_summary(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise RuntimeError('controlled solver failure')
    monkeypatch.setattr(verification,'run_single',fail)
    monkeypatch.setattr(verification,'plot_convergence',lambda *a,**k:None)
    with pytest.raises(ValueError,match='failed/nonfinite'):
        verification.run_full_verification(SimConfig(),'failed',tmp_path,[10,20,40])
    summary=json.loads((tmp_path/'verification/failed/verification_summary.json').read_text())
    assert not summary['passed'] and len(summary['failures'])==2


def test_time_refinement_uses_finest_requested_spatial_budget(tmp_path,monkeypatch):
    calls=[]
    def single(config,*args):
        calls.append((config.numerical.Nx,config.numerical.dt))
        return dict(AI=1.,UI=0.,mean_h=1e-7,effective_dt=config.numerical.dt,
                    maximum_dx=1/config.numerical.Nx,minimum_dx=1/config.numerical.Nx,minimum_dy=1/config.numerical.Ny)
    monkeypatch.setattr(verification,'run_single',single)
    monkeypatch.setattr(verification,'plot_convergence',lambda *a,**k:None)
    verification.run_full_verification(SimConfig(),'fine',tmp_path,[10,20,40])
    assert [nx for nx,dt in calls]==[10,20,40,40,40,40]
    assert len({dt for nx,dt in calls[-3:]})==3


@pytest.mark.parametrize('sizes', [[10,20],[10,10,20],[20,10,40],[2,10,20]])
def test_invalid_spatial_studies_are_rejected(tmp_path,sizes):
    with pytest.raises(ValueError):verification.run_mesh_refinement(SimConfig(),'invalid',tmp_path,sizes)
