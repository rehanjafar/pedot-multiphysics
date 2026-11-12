"""Drive-event timing, administrative censoring and reproducible batch workflows."""
import json
from dataclasses import asdict,replace
import numpy as np
import pandas as pd
import pytest
from pedot_sim.dendrites import DendriteConfig,dendrite_time_grid,simulate_dendrites,initial_electrodes,solve_network_potential
from pedot_sim.dendrite_batch import run_dendrite_batch,summarize_bridging
from pedot_sim.waveforms import pulsed


@pytest.mark.parametrize('period,duty', [(40.,.5),(1.,.01),(.13,.333),(2.,.99)])
def test_direct_drive_resolves_edges_and_duty_fraction(period,duty):
    c=DendriteConfig(steps=4,period_steps=period,duty_cycle=duty)
    t=dendrite_time_grid(c)
    assert t[0]==0 and t[-1]==c.steps and np.all(np.diff(t)>0)
    assert np.max(np.diff(t))<=period/40*(1+1e-12)
    # Integrate an integer number of complete periods using midpoint values.
    cycles=int(c.steps/period)
    cutoff=cycles*period
    keep=t[1:]<=cutoff+1e-12
    if cycles:
        v=np.array([pulsed((a+b)/2,0,1,1/period,duty) for a,b in zip(t[:-1],t[1:])])
        assert (np.diff(t)[keep]*v[keep]).sum()==pytest.approx(cutoff*(2*duty-1),abs=1e-11)


def tiny(**kwargs):
    return replace(DendriteConfig(nx=13,ny=9,wire_length=3,n_particles=5,steps=2,
                                 snapshot_stride=1),**kwargs)


def test_fast_waveform_duration_snapshot_timestamps_and_conservation():
    c=tiny(period_steps=.4,duty_cycle=.03,integration_dt=.4)
    r=simulate_dendrites(c)
    assert r['metrics']['elapsed_model_steps']==c.steps
    assert len(r['records'])>=200
    assert r['snapshot_times'][0]==0 and r['snapshot_times'][-1]==2
    assert len(r['snapshot_times'])==len(r['snapshots'])
    assert r['metrics']['deposited_cells']+len(r['mobile_positions'])==c.n_particles


def test_short_circuit_is_rejected_by_potential_solver():
    owner=initial_electrodes(tiny())
    owner[4,:7]=1;owner[4,7:]=2
    with pytest.raises(ValueError,match='short'):solve_network_potential(owner,1.)


@pytest.mark.parametrize('integration_dt', [0.,-1.,float('nan')])
def test_invalid_integration_intervals(integration_dt):
    with pytest.raises(ValueError):tiny(integration_dt=integration_dt).validate()


@pytest.mark.parametrize('bridged,elapsed', [([False,False],[10.,10.]),([True,True],[2.,4.]),([True,False],[2.,10.])])
def test_bridge_probability_and_restricted_time(bridged,elapsed):
    df=pd.DataFrame({'steps':[10,10],'seed':[1,2],'bridged':bridged,'elapsed_model_steps':elapsed})
    row=summarize_bridging(df,['steps']).iloc[0]
    p=sum(bridged)/2
    assert row.bridge_probability==p
    assert row.bridge_probability_95_low<=p<=row.bridge_probability_95_high
    assert row.restricted_mean_time_model_steps==np.mean(elapsed)
    assert row.censored_count==2-sum(bridged)
    if sum(bridged):assert row.conditional_mean_bridge_time_model_steps==np.mean(np.array(elapsed)[bridged])
    else:assert pd.isna(row.conditional_mean_bridge_time_model_steps)


def test_early_stopping_cannot_masquerade_as_no_bridge():
    df=pd.DataFrame({'steps':[10],'bridged':[False],'elapsed_model_steps':[2.]})
    with pytest.raises(ValueError,match='planned horizon'):summarize_bridging(df,['steps'])


def test_real_batch_resume_provenance_and_analysis(tmp_path):
    c=tiny(steps=1)
    report=run_dendrite_batch(c,'ensemble',tmp_path,[2.,4.],2)
    root=report.parent
    plan=json.loads((root/'plan.json').read_text())
    assert len({cfg['seed'] for cfg in plan['runs'].values()})==4
    assert len(pd.read_csv(root/'bridging_summary.csv'))==2
    metrics=next(root.glob('period*/metrics.json'));mtime=metrics.stat().st_mtime_ns
    assert run_dendrite_batch(c,'ensemble',tmp_path,[2.,4.],2)==report
    assert metrics.stat().st_mtime_ns==mtime
    with pytest.raises(ValueError,match='differs'):run_dendrite_batch(c,'ensemble',tmp_path,[2.,3.],2)
    assert not (root/'campaign.lock').exists()
    params=next(root.glob('period*/params.json'));payload=json.loads(params.read_text());payload['seed']+=1;params.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='mismatched'):run_dendrite_batch(c,'ensemble',tmp_path,[2.,4.],2)


@pytest.mark.parametrize('periods,reps', [([],1),([0],1),([2,2],1),([float('inf')],1),([1],0),([1],True)])
def test_invalid_batch_requests(tmp_path,periods,reps):
    with pytest.raises(ValueError):run_dendrite_batch(tiny(),'bad',tmp_path,periods,reps)


def test_batch_lock_prevents_concurrent_overwrite(tmp_path):
    root=tmp_path/'dendrites/locked';root.mkdir(parents=True);(root/'campaign.lock').touch()
    with pytest.raises(FileExistsError):run_dendrite_batch(tiny(),'locked',tmp_path,[2.],1)
    assert (root/'campaign.lock').exists()


def test_batch_command_line(tmp_path):
    import subprocess,sys
    path=tmp_path/'params.json';path.write_text(json.dumps(asdict(tiny(steps=1))))
    result=subprocess.run([sys.executable,'-m','pedot_sim.cli','run-dendrite-batch','--config',str(path),
        '--campaign-id','cli','--output-base',str(tmp_path),'--periods','4','8','--replicates','2'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert (tmp_path/'dendrites/cli/bridging_summary.csv').exists()
