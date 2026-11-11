"""Input boundaries, statistics, measurement comparisons and full output paths."""
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from pedot_sim.config import SimConfig, validate_config, load_config, save_config
from pedot_sim.runner_single import run_single
from pedot_sim.runner_batch import generate_dense_runs, aggregate_results
from pedot_sim.waveforms import integration_times, get_waveform
from pedot_sim.analysis_stats import _normalize_columns, run_anova, run_regression, run_full_analysis, compute_descriptive_stats
from pedot_sim.analysis_opt import compute_pareto_front
from pedot_sim.validation import align_and_compare, compare_measurements
from pedot_sim.utils_io import save_json, safe_output_path
from pedot_sim.reporting import ReportGenerator


def quick():
    c=SimConfig(); c.numerical.Nx=c.numerical.Ny=31
    c.numerical.t_end=.03; c.numerical.n_snapshots=1; c.output.dpi=30
    return c


@pytest.mark.parametrize('section,name,value', [
    ('numerical','Nx',3.5),('numerical','Ny',True),('numerical','dt',0.),
    ('numerical','dt',float('nan')),('numerical','adaptive_dt',True),
    ('numerical','implicit_theta',1.1),('physics','film_density',0.),
    ('physics','deposition_efficiency',1.1),('physics','deposited_molar_mass',-1.),
    ('physics','conductivity',0.),('physics','T',-1.),('physics','z_species',.5),
    ('waveform','V_AC',-1.),('waveform','frequency',0.),('waveform','waveform_type','typo'),
    ('batch','freq_range',[1.,float('inf')]),('batch','D_range',[2.,1.]),
    ('batch','frequencies',[]),('batch','n_replicates',0),('output','save_fields','false'),
])
def test_invalid_configuration(section,name,value):
    c=quick(); setattr(getattr(c,section),name,value)
    with pytest.raises((ValueError,TypeError)):
        validate_config(c)


def test_json_rejects_unknown_sections_and_roundtrips(tmp_path):
    file=tmp_path/'config.json'; save_config(quick(),file)
    assert load_config(file)==quick()
    file.write_text('{"physic": {"j0": 5}}')
    with pytest.raises(ValueError,match='unknown'):
        load_config(file)


@pytest.mark.parametrize('kind', ['sinusoidal','pulsed','asymmetric_adc'])
def test_time_grid_hits_rest_and_pulse_boundaries(kind):
    c=quick(); c.physics.physics_stage='B'; c.waveform.waveform_type=kind
    c.waveform.frequency=7.; c.waveform.rest_fraction=.173; c.waveform.duty_cycle=.143
    c.numerical.t_end=.43; c.numerical.dt=.1
    endpoints=integration_times(c)
    assert endpoints[0]==0 and endpoints[-1]==.43
    assert np.all(np.diff(endpoints)>0)
    assert np.max(np.diff(endpoints)) <= 1/280+1e-14
    assert np.min(abs(endpoints-(1-.173)/7))<1e-14
    if kind=='pulsed': assert np.min(abs(endpoints-.143/7))<1e-14
    assert get_waveform(c.waveform)(.9/7)==0.


@pytest.mark.parametrize('kind', ['latin_hypercube','random','grid'])
def test_dense_sampling_reproducibility_ranges_and_config_seed(kind):
    c=quick(); c.nucleation.seed=314
    a=generate_dense_runs(c,27,sampling=kind)
    b=generate_dense_runs(c,27,sampling=kind)
    assert a==b
    for _, cfg in a:
        assert c.batch.freq_range[0]<=cfg.waveform.frequency<=c.batch.freq_range[1]
        assert c.batch.D_range[0]*(1-1e-14)<=cfg.physics.diffusion_coefficient<=c.batch.D_range[1]*(1+1e-14)
    if kind!='grid':
        for attr, bounds in [('V_DC',c.batch.V_DC_range),('V_AC',c.batch.V_AC_range)]:
            vals=[getattr(cfg.waveform,attr) for _,cfg in a]
            assert min(vals)>=bounds[0] and max(vals)<=bounds[1]
            if kind=='latin_hypercube':
                bins=np.floor((np.array(vals)-bounds[0])/(bounds[1]-bounds[0])*27).astype(int)
                assert len(set(bins))==27


@pytest.mark.parametrize('stage',['A','B','C','D'])
def test_complete_run_material_audit_and_final_state(stage,tmp_path):
    c=quick(); c.physics.physics_stage=stage; c.physics.deposition_efficiency=.65
    c.numerical.n_snapshots=0; c.output.save_fields=False
    with patch('pedot_sim.runner_single._try_generate_figures'):
        metrics=run_single(c,stage,tmp_path,'test')
    assert 'error' not in metrics
    directory=tmp_path/'simulated_raw/test'/stage
    assert (directory/'final_state.npz').exists()
    audit=json.loads((directory/'research_audit.json').read_text())
    assert audit['material_balance_passed']
    assert audit['experimental_validation']=='not_established'
    assert abs(metrics['mass_balance_error_mol_per_m'])<1e-12
    assert metrics['retained_repeat_mol_per_m']==pytest.approx(.65*metrics['consumed_mol_per_m'])
    series=pd.read_csv(directory/'timeseries.csv')
    assert series.time.iloc[0]==0. and series.time.iloc[-1]==.03
    assert np.all(np.diff(series.time)>0)
    assert metrics['electrolyte_ohmic_energy_J_per_m'] is None if stage=='A' else metrics['electrolyte_ohmic_energy_J_per_m']>0


def test_analysis_runs_for_single_deterministic_condition(tmp_path):
    c=quick()
    with patch('pedot_sim.runner_single._try_generate_figures'):
        run_single(c,'one',tmp_path,'case')
    aggregate_results('case',tmp_path)
    result=run_full_analysis('case',tmp_path)
    assert result['anova']['status']=='not_applicable'
    assert result['regression']['status']=='not_applicable'
    html=ReportGenerator('case',tmp_path).render_html().read_text()
    assert 'not_established' in html and 'not_applicable' in html


def design():
    rng=np.random.default_rng(7); n=50
    df=pd.DataFrame({'frequency':10**rng.uniform(0,2,n),'D':10**rng.uniform(-10,-9,n),
                     'bridge_width':rng.uniform(.001,.004,n),'V_DC':rng.uniform(-1,1,n),
                     'V_AC':rng.uniform(.5,2,n)})
    df['AI']=3+.2*np.log10(df.frequency)+.1*df.V_DC
    df['UI']=1e-7+1e-8*np.log10(df.frequency)
    return df


def test_regression_predicts_independent_held_out_conditions():
    df=design(); result=run_regression(df)
    assert result['AI_r_squared']==pytest.approx(1.)
    assert result['AI_cv_rmse']<1e-12
    assert result['UI_cv_rmse']<1e-18
    assert 'V_DC' in result['AI_coefficients']
    assert 'bridge_width' in result['AI_coefficients']
    assert (_normalize_columns(df).D_level=='continuous').all()


def test_anova_rejects_deterministic_pseudoreplicates():
    from itertools import product
    df=pd.DataFrame(product([1,10],['low','high'],['narrow','wide'],[1,2]),columns=['frequency','D_level','geometry','seed'])
    df['AI']=1.;df['UI']=0.;df['noise_enabled']=False
    with pytest.raises(ValueError,match='deterministic'):
        run_anova(df)


def test_continuous_conditions_are_not_grouped_into_fake_replicates():
    df=design(); result=compute_descriptive_stats(df)
    assert len(result)==len(df)
    assert (result.AI_count==1).all()


def test_pareto_known_front_and_nonfinite_rejection():
    df=pd.DataFrame({'UI':[1.,2.,1.,3.],'mean_growth_rate':[1.,2.,.5,.2]})
    assert list(compute_pareto_front(df).index)==[0,1]
    df.loc[0,'UI']=np.nan
    with pytest.raises(ValueError): compute_pareto_front(df)


@pytest.mark.parametrize('case', ['nan','duplicate','descending','outside','empty'])
def test_bad_validation_data_rejected(case):
    sim=pd.DataFrame({'t':[0.,1.,2.],'I':[1.,2.,3.]}); exp=sim.copy()
    if case=='nan': exp.loc[0,'I']=np.nan
    if case=='duplicate': sim.loc[1,'t']=0.
    if case=='descending': sim=sim.iloc[::-1]
    if case=='outside': exp.loc[2,'t']=3.
    if case=='empty': exp=exp.iloc[:0]
    with pytest.raises(ValueError): align_and_compare(sim,exp)


def test_traceable_measurement_comparison_and_threshold(tmp_path):
    sim=tmp_path/'sim.csv'; exp=tmp_path/'exp.csv'; meta=tmp_path/'meta.json'
    pd.DataFrame({'time':[0.,1.,2.],'mean_h':[0.,1e-6,2e-6]}).to_csv(sim,index=False)
    pd.DataFrame({'time':[.5,1.5],'mean_h':[.6e-6,1.6e-6]}).to_csv(exp,index=False)
    meta.write_text(json.dumps({'source':'Synthetic unit-test fixture, not experimental data', 'units':{'time':'s','mean_h':'m'}}))
    result=compare_measurements(sim,exp,meta,tmp_path/'compare',max_rmse=2e-7)
    assert result['RMSE']==pytest.approx(1e-7)
    assert result['bias']==pytest.approx(-1e-7)
    assert result['acceptance']=='passed' and len(result['sha256']['experiment'])==64
    result=compare_measurements(sim,exp,meta,tmp_path/'compare',max_rmse=5e-8)
    assert result['acceptance']=='failed'
    meta.write_text(json.dumps({'source':'synthetic','units':{'time':'s','mean_h':'um'}}))
    with pytest.raises(ValueError,match='units'): compare_measurements(sim,exp,meta,tmp_path/'bad')


def test_atomic_json_and_symlink_escape(tmp_path):
    file=tmp_path/'output.json'; save_json({'zero':0,'undefined':float('inf')},file)
    assert json.loads(file.read_text())=={'zero':0,'undefined':None}
    outside=tmp_path/'outside'; outside.mkdir()
    base=tmp_path/'base'; base.mkdir()
    (base/'escape').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'): safe_output_path(base,'escape','file')
    assert not list(tmp_path.glob('.pedot-*'))


def test_replicated_stochastic_factorial_anova_is_finite():
    from itertools import product
    rng=np.random.default_rng(88)
    df=pd.DataFrame(product([1.,10.],['low','high'],['narrow','wide'],range(5)),columns=['frequency','D_level','geometry','rep'])
    df['seed']=np.arange(len(df)); df['noise_enabled']=True
    df['AI']=2+.03*df.frequency+rng.normal(0,.02,len(df))
    df['UI']=1e-7+df.frequency*1e-9+rng.normal(0,1e-9,len(df))
    result=run_anova(df)
    assert result['AI_anova']['sum_sq'].notna().all()
    assert result['UI_anova']['eta_sq'].between(0,1).all()


def test_real_parallel_batch_resume_and_cli_workflow(tmp_path):
    from pedot_sim.runner_batch import run_batch
    from pedot_sim.cli import main
    c=quick(); c.numerical.t_end=.01; c.batch.frequencies=[1.,2.]
    c.batch.D_levels=['mid']; c.batch.geometries=['narrow'];c.batch.n_replicates=1
    result=run_batch(c,'parallel',tmp_path,max_workers=2)
    assert len(result)==2
    repeated=run_batch(c,'parallel',tmp_path,max_workers=1)
    assert set(repeated.run_id)==set(result.run_id)
    for command in ['analyze','make-report']:
        with pytest.raises(SystemExit) as exc:
            main([command,'--campaign-id','parallel','--output-base',str(tmp_path)])
        assert exc.value.code==0
    assert (tmp_path/'reports/parallel/final_report.html').exists()
