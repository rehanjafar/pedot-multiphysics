"""Geometry feedback and aggregate-conservation tests for direct-wire growth."""
import json
from dataclasses import asdict
import numpy as np
import pytest
from pedot_sim.dendrites import (DendriteConfig, initial_electrodes,
    solve_network_potential, simulate_dendrites, run_dendrite, analyze_dendrite_campaign)


def small(**changes):
    config = DendriteConfig(nx=25, ny=17, wire_length=4, wire_width=3,
                            n_particles=180, steps=30, snapshot_stride=10,
                            attachment_probability=1.)
    for key, value in changes.items():
        setattr(config, key, value)
    return config


@pytest.mark.parametrize('key,value', [('nx',6), ('wire_width',2), ('wire_length',12),
    ('n_particles',5000), ('steps',0), ('period_steps',0), ('amplitude_V',-1),
    ('attachment_probability',1.1), ('mobility_cells2_per_V_step',-1),
    ('activation_voltage_V',-1), ('seed',-1), ('duty_cycle',1), ('offset_V',float('nan'))])
def test_bad_dendrite_parameters(key, value):
    with pytest.raises(ValueError):
        small(**{key:value}).validate()


@pytest.mark.parametrize('voltage', [-5., 0., 5.])
def test_all_conducting_cells_are_equipotential(voltage):
    owner = initial_electrodes(small())
    owner[8, 4:8] = 1  # A conducting PEDOT extension, not only the initial wire
    V, residual = solve_network_potential(owner, voltage)
    np.testing.assert_array_equal(V[owner==1], voltage)
    np.testing.assert_array_equal(V[owner==2], 0.)
    assert residual < 1e-10
    assert V.min() >= min(voltage, 0.)-1e-10
    assert V.max() <= max(voltage, 0.)+1e-10


def test_deposit_geometry_changes_solution_field():
    owner = initial_electrodes(small())
    before, _ = solve_network_potential(owner, 5.)
    owner[8, 4:10] = 1
    after, _ = solve_network_potential(owner, 5.)
    assert np.linalg.norm(after-before) > 1.
    assert after[8, 9] == 5.


@pytest.mark.parametrize('seed', [2, 7, 19])
def test_growth_conserves_all_aggregates_and_connects_deposits(seed):
    from scipy.ndimage import label
    config = small(seed=seed)
    result = simulate_dendrites(config)
    assert result['metrics']['deposited_cells'] > 0
    assert result['metrics']['deposited_cells']+len(result['mobile_positions']) == config.n_particles
    for network in (1,2):
        _, count = label(result['owner'] == network)
        assert count == 1
    pos = result['mobile_positions']
    assert np.all(result['owner'][pos[:,0],pos[:,1]] == 0)
    assert len(np.unique(pos,axis=0)) == len(pos)
    counts = np.count_nonzero(result['snapshots'],axis=(1,2))
    assert np.all(np.diff(counts)>=0)


@pytest.mark.parametrize('changes', [{'amplitude_V':0.}, {'attachment_probability':0.}, {'activation_voltage_V':100.}])
def test_no_attachment_without_drive_or_activation(changes):
    result = simulate_dendrites(small(**changes))
    assert result['metrics']['deposited_cells'] == 0
    np.testing.assert_array_equal(result['owner'], result['initial_owner'])


def test_seed_reproducibility_and_different_seed_variability():
    first = simulate_dendrites(small(seed=17))
    second = simulate_dendrites(small(seed=17))
    third = simulate_dendrites(small(seed=18))
    np.testing.assert_array_equal(first['owner'],second['owner'])
    assert not np.array_equal(first['owner'],third['owner'])


def test_growth_stops_at_electrical_contact():
    config = small(nx=13, ny=9, wire_length=4, n_particles=93,
                    steps=300, amplitude_V=10., activation_voltage_V=0.)
    result = simulate_dendrites(config)
    assert result['metrics']['bridged']
    assert result['metrics']['elapsed_model_steps'] < config.steps
    assert result['records'][-1]['bridged']
    assert sum(r['bridged'] for r in result['records']) == 1


def test_direct_wire_output_and_campaign_report(tmp_path):
    for seed in [1,2]:
        result = run_dendrite(small(steps=5, seed=seed), f'r{seed}', tmp_path, 'wires')
        assert result['deposited_cells'] > 0
    report = analyze_dendrite_campaign('wires',tmp_path)
    assert report.exists()
    assert 'No experimental validation' in report.read_text()
    metadata = json.loads((tmp_path/'dendrites/wires/r1/metadata.json').read_text())
    assert metadata['time_unit']=='model_step'
    assert metadata['experimental_validation']=='not_established'
    with pytest.raises(FileExistsError):
        run_dendrite(small(), 'r1', tmp_path,'wires')
