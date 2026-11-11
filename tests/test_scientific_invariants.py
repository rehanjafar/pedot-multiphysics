"""Numerical invariants and regression cases from the repository audit."""
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import pytest
from pedot_sim.config import SimConfig, GeometryConfig
from pedot_sim.geometry import build_domain
from pedot_sim.transport import solve_diffusion_step, add_migration_flux
from pedot_sim.growth import evolve_one_timestep
from pedot_sim.field import compute_electric_field, get_overpotential_at_bpe
from pedot_sim.kinetics import butler_volmer_flux
from pedot_sim.metrics import anisotropy_index
from pedot_sim.runner_single import run_single
from pedot_sim.runner_batch import generate_factorial_runs, generate_dense_runs, _run_worker_direct, _config_to_dict, run_batch
from pedot_sim.utils_grid import create_grid
from pedot_sim.utils_io import validate_id, get_campaign_dir
from pedot_sim.verification import _richardson_error
from pedot_sim.reporting import ReportGenerator


def config():
    c = SimConfig(); c.numerical.Nx = c.numerical.Ny = 30
    c.numerical.dt = .01; c.numerical.t_end = .03; c.numerical.n_snapshots = 2
    c.output.dpi = 30
    return c


@pytest.mark.parametrize('theta', [0., .5, 1.])
def test_insulated_uniform_field(theta):
    c = config(); d = build_domain(c.geometry, c.numerical)
    C = np.where(d.domain_mask, 50., 0.)
    out = solve_diffusion_step(C, d.grid, 6.5e-10, .01, np.zeros_like(C),
        d.domain_mask, 50., d.left_feeder_mask, d.right_feeder_mask, theta)
    np.testing.assert_allclose(out, C, atol=1e-12)


def test_closed_diffusion_matches_eigenmode_and_conserves():
    g = create_grid(1., 1., 15, 15); mask = np.ones((15, 15), bool); zero = np.zeros_like(mask)
    mode = np.cos(np.pi*(np.arange(15)+.5)/15)[None, :] * np.ones((15, 1))
    C = 2. + mode; dt = .1; D = .2
    actual = solve_diffusion_step(C, g, D, dt, np.zeros_like(C), mask, 2., zero, zero, 1.)
    eigenvalue = -4*D*np.sin(np.pi/30)**2 / g.dx**2
    np.testing.assert_allclose(actual, 2.+mode/(1-dt*eigenvalue), atol=1e-12)
    assert actual.sum() == pytest.approx(C.sum())
    assert actual.min() >= C.min() and actual.max() <= C.max()


def test_migration_conservative_positive():
    c = config(); d = build_domain(c.geometry, c.numerical)
    C = np.where(d.domain_mask, 50., 0.); V = d.grid.X * 1000
    out = add_migration_flux(C, V, d.grid, 1e-8, 1, .1, d.domain_mask)
    assert out.sum() == pytest.approx(C.sum()); assert out.min() >= 0
    np.testing.assert_allclose(add_migration_flux(C, V, d.grid, 1e-8, 0, .1, d.domain_mask), C)


def test_constant_masked_potential_has_no_artificial_field():
    c=config(); d=build_domain(c.geometry,c.numerical)
    Ex,Ey=compute_electric_field(d.domain_mask.astype(float),d.grid,d.domain_mask)
    assert not np.any(Ex); assert not np.any(Ey)


@pytest.mark.parametrize('N',[30,60])
def test_film_mass_equals_consumption(N):
    c=config();c.numerical.Nx=c.numerical.Ny=N;c.physics.diffusion_coefficient=0.;c.physics.k_rxn=100.
    d=build_domain(c.geometry,c.numerical); C=np.where(d.domain_mask,50.,0.);h=np.zeros(d.electrode_mask.sum())
    result=evolve_one_timestep(C,h,0.,1.,c,d,None)
    loss=(C-result.C)[d.electrode_mask]*d.grid.dy
    np.testing.assert_allclose(result.h*1340/.14016,loss)
    assert np.all(result.C>=0);assert np.all(loss<=50*d.grid.dy)


def test_zero_monomer_cannot_create_film_even_with_noise():
    from pedot_sim.nucleation import NucleationModel
    c=config();c.physics.diffusion_coefficient=0.;c.nucleation.enable_noise=True;c.nucleation.nucleation_rate=1e7
    d=build_domain(c.geometry,c.numerical);C=np.zeros(d.domain_mask.shape);h=np.zeros(d.electrode_mask.sum())
    result=evolve_one_timestep(C,h,0.,.01,c,d,NucleationModel(c.nucleation,len(h)))
    np.testing.assert_array_equal(result.h,h)


def test_bv_floating_balance_and_anodic_only():
    from pedot_sim.transport import compute_reaction_rate
    c=config();c.physics.physics_stage='D'; V=np.array([[-.1,0.,.1]])
    mask=np.ones_like(V,dtype=bool);C=np.array([[50.,50.,5.]])
    eta=get_overpotential_at_bpe(V,mask,C,c.physics)
    j=butler_volmer_flux(eta,c.physics.j0,C[mask],50.)
    assert abs(j.sum()) < 1e-8
    flux=compute_reaction_rate(C,mask,c.physics,eta=eta)[mask]
    np.testing.assert_array_equal(flux[j<0],0.)


def test_ai_includes_bare_electrode():
    assert anisotropy_index(np.array([0.,0.,1.]))==3.


def test_sweep_preserves_width_and_fractional_identity(tmp_path):
    c=config();c.batch.frequencies=[1.1,1.9];c.batch.D_levels=['mid'];c.batch.geometries=['narrow'];c.batch.n_replicates=1
    runs=generate_factorial_runs(c);assert len(runs)==len(set(rid for rid,_ in runs))==2
    _,sample=generate_dense_runs(c,n_runs=1)[0]
    assert GeometryConfig(**asdict(sample.geometry)).bridge_width==sample.geometry.bridge_width
    with patch('pedot_sim.runner_single._try_generate_figures'):
        _run_worker_direct('worker',_config_to_dict(sample),str(tmp_path),'sweep')
    params=json.loads((tmp_path/'simulated_raw/sweep/worker/params.json').read_text())
    assert params['geometry']['bridge_width']==sample.geometry.bridge_width


def test_time_and_growth_independent_of_snapshots(tmp_path):
    c=config();c.numerical.dt=.1;c.numerical.t_end=.25;c.numerical.n_snapshots=0;c.output.save_fields=False
    with patch('pedot_sim.runner_single._try_generate_figures'):
        m=run_single(c,'time',tmp_path,'case')
    assert m['mean_growth_rate']>0
    directory=tmp_path/'simulated_raw/case/time'
    assert not (directory/'fields.npz').exists()
    assert json.loads((directory/'metadata.json').read_text())['final_time']==pytest.approx(.25)
    assert pd.read_csv(directory/'timeseries.csv').time.iloc[-1]==pytest.approx(.25)
    assert m['timescale_ratio']==pytest.approx(c.geometry.electrode_length*c.physics.k_rxn/c.physics.diffusion_coefficient)


def test_waveform_sampling_cap(tmp_path):
    c=config();c.physics.physics_stage='B';c.waveform.frequency=100.;c.numerical.dt=.01;c.numerical.t_end=.001
    with patch('pedot_sim.runner_single._try_generate_figures'):
        m=run_single(c,'wave',tmp_path,'case')
    assert m['effective_dt'] <= 1/4000
    assert 'error' not in m


@pytest.mark.parametrize('name',['../escape','/tmp/escape','x/y','<script>'])
def test_unsafe_identifiers_rejected(name):
    with pytest.raises(ValueError):validate_id(name)


def test_report_escapes_filename_caption(tmp_path):
    gen=ReportGenerator('safe',tmp_path)
    (gen.fig_dir/'<img src=x onerror=alert(1)>.png').touch()
    html=gen.render_html().read_text()
    assert '<img src=x onerror=alert(1)>' not in html
    assert '&lt;img' in html


def test_richardson_uses_actual_refinement_and_observed_order():
    h=np.array([.1,.05,.01]);values=3+2*h
    error=_richardson_error(values,h)
    assert error[-1]==pytest.approx(.02)
    assert np.isnan(_richardson_error(np.array([1.,2.,1.]),h)).all()


def test_manifest_location_and_changed_config_refusal(tmp_path):
    c=config();directory=get_campaign_dir(tmp_path,'case')
    assert directory==tmp_path/'simulated_agg/case'
    # Avoid a process pool: simulate an already completed campaign with no pending runs.
    with patch('pedot_sim.runner_batch.generate_factorial_runs',return_value=[]):
        run_batch(c,'case',tmp_path)
        c.physics.k_rxn*=2
        with pytest.raises(ValueError,match='changed'):run_batch(c,'case',tmp_path)


def test_verification_rejects_unconverged_mesh(tmp_path):
    from pedot_sim.verification import run_full_verification
    mesh=pd.DataFrame({'N':[50,100,200], 'AI':[1.,1.1,1.3], 'UI':[0.,0.,0.], 'mean_h':[1.,1.,1.]})
    time=pd.DataFrame({'dt':[.01,.005,.001], 'effective_dt':[.01,.005,.001], 'AI':[1.,1.,1.], 'UI':[0.,0.,0.], 'mean_h':[1.,1.,1.]})
    with patch('pedot_sim.verification.run_mesh_refinement',return_value=mesh), patch('pedot_sim.verification.run_timestep_refinement',return_value=time), patch('pedot_sim.verification.plot_convergence'):
        with pytest.raises(ValueError,match='mesh refinement'):
            run_full_verification(config(),'check',tmp_path)


def test_cli_accepts_readme_verbosity_and_reports_failures(tmp_path):
    from pedot_sim.cli import main
    with patch('pedot_sim.cli._COMMAND_DISPATCH', {'run-single': lambda args: 0}):
        with pytest.raises(SystemExit) as result:
            main(['run-single','--config','example.json','--run-id','one','-v'])
        assert result.value.code == 0
    from pedot_sim.cli import _cmd_verify
    from types import SimpleNamespace
    args=SimpleNamespace(config=Path('unused'),output_base=str(tmp_path),campaign_id='verify')
    with patch('pedot_sim.cli._load_config',return_value=config()), patch('pedot_sim.verification.run_full_verification',side_effect=ValueError('not converged')):
        assert _cmd_verify(args)==1


def test_native_flux_is_not_silently_compared_to_amperes(tmp_path):
    from pedot_sim.validation import run_validation
    sim=tmp_path/'sim.csv';exp=tmp_path/'exp.csv'
    pd.DataFrame({'time':[0.,1.], 'flux':[1.,2.]}).to_csv(sim,index=False)
    pd.DataFrame({'t':[0.,1.], 'I':[1.,2.]}).to_csv(exp,index=False)
    report=run_validation(sim,exp,tmp_path/'report')
    assert report.metrics['status']=='failed'
    assert 'dimensioned I' in report.metrics['error']
