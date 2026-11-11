"""Finite-volume identities and independent analytic benchmarks on stretched grids."""
from types import SimpleNamespace
import numpy as np
import pytest
from pedot_sim.config import SimConfig
from pedot_sim.geometry import build_domain
from pedot_sim.utils_grid import create_refined_grid, cell_areas, electrode_weights
from pedot_sim.transport import _diffusion_operator, solve_diffusion_step, reservoir_exchange, add_migration_flux
from pedot_sim.field import get_overpotential_at_bpe
from pedot_sim.kinetics import butler_volmer_flux
from pedot_sim.research import ohmic_power_per_depth


@pytest.mark.parametrize('resolution,stretch', [(30,2.), (60,6.), (100,10.)])
def test_exact_geometry_and_weighted_diffusion_conservation(resolution,stretch):
    c=SimConfig();c.numerical.Nx=c.numerical.Ny=resolution;c.numerical.mesh_stretch=stretch
    d=build_domain(c.geometry,c.numerical);g=d.grid
    area=cell_areas(g);geo=c.geometry
    assert area.sum()==pytest.approx(d.Lx*d.Ly)
    assert area[d.domain_mask].sum()==pytest.approx(2*geo.compartment_width*geo.compartment_height+geo.bridge_length*geo.bridge_width)
    assert electrode_weights(g,d.electrode_mask).sum()==pytest.approx(geo.electrode_length)
    L=_diffusion_operator(g,d.domain_mask)
    residual=(area.ravel()@L)
    assert np.max(np.abs(residual))<2e-10
    assert np.max(np.abs(L@np.ones(area.size))/np.maximum(1.,np.abs(L.diagonal())))<1e-15


def test_simultaneous_diffusion_and_local_reaction_two_cell_exact():
    # Two volumes, one face of unit conductance, first-order sink only in cell 0.
    # Backward Euler: [3,-1;-1,2] C_new = [1,1], giving [0.6,0.8].
    g=SimpleNamespace(Nx=2,Ny=1,dx=1.,dy=1.,x=np.array([0.,1.]),y=np.array([0.]))
    initial=np.ones((1,2));mask=np.ones_like(initial,bool);empty=~mask
    c=solve_diffusion_step(initial,g,1.,1.,np.zeros_like(initial),mask,1.,empty,empty,
                           theta=1.,reaction_coefficient=np.array([[1.,0.]]))
    np.testing.assert_allclose(c,[[.6,.8]],atol=1e-14)


@pytest.mark.parametrize('stretch', [1.,4.,8.])
def test_independent_boundary_flux_closes_reaction_inventory(stretch):
    g=create_refined_grid(1.,1.,25,12,[.5],[.5],stretch)
    mask=np.ones((g.Ny,g.Nx),bool);left=np.zeros_like(mask);right=left.copy();left[:,0]=True;right[:,-1]=True
    initial=np.random.default_rng(22).uniform(.2,1.,mask.shape);initial[left|right]=1.
    reaction=np.zeros_like(initial);reaction[:,g.Nx//2]=20.
    final=solve_diffusion_step(initial,g,.03,.1,np.zeros_like(initial),mask,1.,left,right,1.,reaction)
    reservoir=reservoir_exchange(initial,final,g,.03,.1,1.,mask,left|right)
    area=cell_areas(g)
    residual=np.sum((final-initial)*area)+np.sum(.1*reaction*final*area)-reservoir
    assert abs(residual)<1e-11
    assert np.all(final>=0)


@pytest.mark.parametrize('charge', [-1,0,1])
def test_graded_migration_preserves_inventory(charge):
    g=create_refined_grid(1.,1.,20,20,[.5],[.5],3.)
    mask=np.ones((g.Ny,g.Nx),bool);mask[:2,3:6]=False
    initial=np.where(mask,np.random.default_rng(4).uniform(.2,1.,mask.shape),0.)
    final=add_migration_flux(initial,2*g.X-g.Y,g,.001,charge,.03,mask)
    np.testing.assert_allclose(np.sum(final*cell_areas(g)),np.sum(initial*cell_areas(g)),rtol=1e-13)
    assert final.min()>=0
    assert not final[~mask].any()


def test_graded_laplace_and_joule_loss_match_rectangular_solution():
    g=create_refined_grid(1.,2.,35,16,[],[],4.)
    mask=np.ones((g.Ny,g.Nx),bool)
    from pedot_sim.field import _graded_unit_potential
    left=np.zeros_like(mask);right=left.copy();left[:,0]=True;right[:,-1]=True
    V,residual=_graded_unit_potential(mask.shape,mask.tobytes(),left.tobytes(),right.tobytes(),
                                      tuple(g.x),tuple(g.y),tuple(g.widths_x),tuple(g.widths_y))
    expected=.5-(g.X-g.x[0])/(g.x[-1]-g.x[0])
    np.testing.assert_allclose(V,expected,atol=2e-12)
    assert residual<1e-10
    d=SimpleNamespace(grid=g,domain_mask=mask)
    assert ohmic_power_per_depth(V,d,3.)==pytest.approx(3*2/(g.x[-1]-g.x[0]),rel=1e-11)


def test_floating_current_balance_uses_surface_measure():
    c=SimConfig();V=np.array([[0.,.03,.08]]);mask=np.ones_like(V,bool);C=np.array([[50.,25.,45.]])
    area=np.array([.1,2.,.3])
    eta=get_overpotential_at_bpe(V,mask,C,c.physics,area)
    ph=c.physics
    j=butler_volmer_flux(eta,ph.j0,C[mask],ph.C_bulk,ph.alpha,ph.n_electrons,ph.T)
    assert abs(j@area)<1e-8
    assert abs(j.sum())>1


@pytest.mark.parametrize('stage', ['A','B','C','D'])
@pytest.mark.parametrize('noise', [False,True])
def test_graded_end_to_end_material_audits(tmp_path,monkeypatch,stage,noise):
    from pedot_sim.runner_single import run_single
    from pedot_sim.utils_io import load_json,load_npz
    c=SimConfig();c.physics.physics_stage=stage;c.physics.deposition_efficiency=.3
    c.numerical.Nx=40;c.numerical.Ny=32;c.numerical.mesh_stretch=6.
    c.numerical.t_end=.02;c.numerical.dt=.01;c.numerical.n_snapshots=0
    c.physics.z_species=1 if stage in ('C','D') else 0
    c.nucleation.enable_noise=noise
    monkeypatch.setattr('pedot_sim.runner_single._try_generate_figures',lambda *a,**k:None)
    result=run_single(c,'r',tmp_path,'audit')
    assert result.get('status')!='failed',result
    root=tmp_path/'simulated_raw/audit/r'
    assert load_json(root/'research_audit.json')['material_balance_passed']
    state=load_npz(root/'final_state.npz')
    assert state['C'].min()>=0
    assert abs(result['mass_balance_error_mol_per_m'])<1e-12
    assert result['retained_repeat_mol_per_m']==pytest.approx(.3*result['consumed_mol_per_m'],rel=1e-12)
