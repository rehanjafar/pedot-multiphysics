"""Independent analytic benchmarks; these are numerical, not empirical validation."""
import numpy as np
import pytest

from pedot_sim.config import SimConfig
from pedot_sim.field import solve_laplace_2d, fixed_domain_potential, get_overpotential_at_bpe
from pedot_sim.geometry import build_domain
from pedot_sim.growth import evolve_one_timestep
from pedot_sim.kinetics import butler_volmer_flux, effective_overpotential, flux_to_consumption_rate
from pedot_sim.metrics import depletion_depth
from pedot_sim.transport import solve_diffusion_step, add_migration_flux, compute_reaction_rate
from pedot_sim.utils_grid import create_grid
from pedot_sim.research import ohmic_power_per_depth


def rectangle(n=25):
    grid = create_grid(1., .2, n, 5)
    mask = np.ones((5, n), bool)
    left, right = np.zeros_like(mask), np.zeros_like(mask)
    left[:, 0] = True
    right[:, -1] = True
    return grid, mask, left, right


@pytest.mark.parametrize('drive', [-7., 0., .1, 3.])
def test_laplace_matches_exact_rectangular_solution(drive):
    grid, mask, left, right = rectangle()
    V, _, residual = solve_laplace_2d(grid, mask, drive/2, -drive/2,
                                     left, right, np.zeros_like(mask), tol=1e-10)
    np.testing.assert_allclose(V, np.broadcast_to(drive*(.5-grid.x), V.shape), atol=1e-9)
    assert residual < 1e-10


def diffuse(n, dt, duration, theta):
    grid, mask, left, right = rectangle(n)
    C = 1+np.broadcast_to(np.sin(np.pi*grid.x), mask.shape).copy()
    for _ in range(round(duration/dt)):
        C = solve_diffusion_step(C, grid, .1, dt, np.zeros_like(C), mask, 1., left, right, theta)
    return C, grid


def test_second_order_spatial_convergence_to_continuum_diffusion():
    errors = []
    for n in [17, 33, 65]:
        actual, grid = diffuse(n, .0002, .02, .5)
        exact = 1+np.exp(-.1*np.pi**2*.02)*np.sin(np.pi*grid.x)
        errors.append(np.max(np.abs(actual-exact)))
    assert 3.8 < errors[0]/errors[1] < 4.2
    assert 3.8 < errors[1]/errors[2] < 4.2


@pytest.mark.parametrize('theta,order', [(1., 1), (.5, 2)])
def test_temporal_order_against_exact_discrete_mode(theta, order):
    errors = []
    for dt in [.02, .01, .005]:
        actual, grid = diffuse(33, dt, .1, theta)
        eigenvalue = -4*.1*np.sin(np.pi/(2*32))**2/grid.dx**2
        exact = 1+np.exp(eigenvalue*.1)*np.sin(np.pi*grid.x)
        errors.append(np.max(np.abs(actual-exact)))
    assert 2**order*.9 < errors[-2]/errors[-1] < 2**order*1.1


@pytest.mark.parametrize('seed', [3, 11, 41])
def test_masked_diffusion_conserves_random_closed_inventory(seed):
    grid, mask, _, _ = rectangle()
    rng = np.random.default_rng(seed)
    mask[1:4, 9:12] = False
    C = np.where(mask, rng.uniform(.01, 2., mask.shape), 0.)
    zero = np.zeros_like(mask)
    actual = solve_diffusion_step(C, grid, .3, .2, np.zeros_like(C), mask, 1., zero, zero, 1.)
    assert actual.sum() == pytest.approx(C.sum(), rel=1e-12)
    assert np.all(actual[~mask] == 0)
    assert actual[mask].min() >= C[mask].min()
    assert actual.max() <= C.max()


@pytest.mark.parametrize('z', [-2, -1, 0, 1, 2])
def test_migration_direction_and_inventory(z):
    grid, mask, _, _ = rectangle()
    C = np.ones(mask.shape)
    V = np.broadcast_to(grid.x, mask.shape)
    result = add_migration_flux(C, V, grid, .001, z, .1, mask)
    assert result.min() >= 0
    assert result.sum() == pytest.approx(C.sum())
    displacement = np.sum((result-C)*grid.X)
    if z:
        assert displacement*z < 0  # cations drift toward lower solution potential
    else:
        np.testing.assert_array_equal(result, C)


@pytest.mark.parametrize('stage', ['A','B','C','D'])
@pytest.mark.parametrize('efficiency', [0., .35, 1.])
def test_single_step_balances_film_and_consumption(stage, efficiency):
    c = SimConfig(); c.numerical.Nx = c.numerical.Ny = 31
    c.physics.physics_stage = stage
    c.physics.deposition_efficiency = efficiency
    c.physics.film_density = 1500.
    d = build_domain(c.geometry, c.numerical)
    concentration = np.where(d.domain_mask, c.physics.C_bulk, 0.)
    h = np.zeros(np.sum(d.electrode_mask))
    result = evolve_one_timestep(concentration, h, 0., .001, c, d, None)
    consumed = np.sum(result.R_field)*d.grid.dx*d.grid.dy*.001
    retained = np.sum(result.h)*d.grid.dx*1500/.14016
    assert retained == pytest.approx(consumed*efficiency, rel=1e-10, abs=1e-20)
    assert np.isfinite(result.C).all() and result.C.min() >= 0
    assert result.total_flux*.001 == pytest.approx(consumed)


def test_floating_voltage_and_overpotential_share_sign_convention():
    c = SimConfig(); c.numerical.Nx = c.numerical.Ny = 31
    c.physics.alpha = .37
    d = build_domain(c.geometry, c.numerical)
    V, _, _ = fixed_domain_potential(d, 2., 1e-8, 10000, 2.)
    concentration = np.where(d.domain_mask, 20+10*d.grid.X/d.Lx, 0.)
    eta = get_overpotential_at_bpe(V, d.electrode_mask, concentration, c.physics)
    metal = eta + V[d.electrode_mask]
    np.testing.assert_allclose(metal, metal[0], atol=1e-12)
    np.testing.assert_allclose(effective_overpotential(V[d.electrode_mask], metal[0]), eta)
    j = butler_volmer_flux(eta, 10., concentration[d.electrode_mask], 50., .37)
    assert abs(j.sum()) < 1e-8*max(1., np.abs(j).sum())


def test_bv_zero_sign_oddness_and_faradaic_units():
    eta = np.array([-.01, 0., .01])
    j = butler_volmer_flux(eta, 2., 50., 50.)
    assert j[1] == 0 and j[0] < 0 < j[2]
    assert j[0] == pytest.approx(-j[2])
    from pedot_sim.constants import F
    np.testing.assert_allclose(flux_to_consumption_rate(np.array([-F, 2*F])), [0., 1.])


def test_depletion_stops_at_insulating_wall():
    c = SimConfig(); c.numerical.Nx = c.numerical.Ny = 31
    d = build_domain(c.geometry, c.numerical)
    depleted = np.zeros_like(d.grid.X)
    depth = depletion_depth(depleted, 50., d.grid, d.electrode_mask, domain_mask=d.domain_mask)
    assert depth <= c.geometry.bridge_width/2 + d.grid.dy


def test_cached_field_linearity_and_joule_scaling():
    c = SimConfig(); c.numerical.Nx = c.numerical.Ny = 31
    d = build_domain(c.geometry, c.numerical)
    v1, _, _ = fixed_domain_potential(d, 1., 1e-8, 10000, 2.)
    v2, _, _ = fixed_domain_potential(d, -2., 1e-8, 10000, 2.)
    np.testing.assert_allclose(v2, -2*v1)
    p1 = ohmic_power_per_depth(v1, d, 1.)
    assert p1 > 0
    assert ohmic_power_per_depth(v2, d, 1.) == pytest.approx(4*p1)
    assert ohmic_power_per_depth(v1, d, 3.) == pytest.approx(3*p1)


@pytest.mark.parametrize('stage', ['B','C','D','unknown'])
def test_missing_physics_inputs_never_fall_back(stage):
    c = SimConfig(); c.physics.physics_stage = stage
    with pytest.raises(ValueError):
        compute_reaction_rate(np.ones((3,3)), np.ones((3,3), bool), c.physics)
