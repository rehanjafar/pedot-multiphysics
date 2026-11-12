"""Analytic accounting, data provenance, units and hostile metadata tests."""
import json
import hashlib
import numpy as np
import pandas as pd
import pytest
from pedot_sim.manufacturing import integrate_signed_product, analyze_manufacturing


@pytest.mark.parametrize('t,v,i,expected', [
    ([0,2],[3,3],[4,4],(24,24,0)),
    ([0,2],[3,3],[2,-2],(0,3,3)),
    ([0,1],[-1,1],[-1,1],(1/3,1/3,0)),
    ([0,1],[-1,1],[1,-1],(-1/3,0,1/3)),
    ([0,1],[0,1],[0,1],(1/3,1/3,0)),
    ([0,1,3],[2,2,2],[1,1,1],(6,6,0)),
])
def test_exact_piecewise_linear_energy(t,v,i,expected):
    np.testing.assert_allclose(integrate_signed_product(t,v,i),expected,atol=1e-14)


@pytest.mark.parametrize('t,v,i', [([0],[1],[1]),([0,0],[1,1],[1,1]),([1,0],[1,1],[1,1]),
    ([0,1],[1],[1,1]),([0,1],[1,float('inf')],[1,1]),([[0,1]],[[1,1]],[[1,1]])])
def test_reject_bad_measurements(t,v,i):
    with pytest.raises(ValueError):integrate_signed_product(t,v,i)


def inputs(tmp_path,**extra):
    trace=tmp_path/'trace.csv';meta=tmp_path/'metadata.json'
    pd.DataFrame({'time_s':[0.,1.,2.],'voltage_V':[2.,2.,2.],'current_A':[.1,.1,.1]}).to_csv(trace,index=False)
    meta.write_text(json.dumps({'source':'synthetic analytic test <script>alert(1)</script>',
        'units':{'time_s':'s','voltage_V':'V','current_A':'A'},**extra}))
    return trace,meta,tmp_path/'analysis'


def test_measured_film_formulas_and_traceability(tmp_path):
    trace,meta,out=inputs(tmp_path,dry_product_mass_kg=1e-6,film_area_m2=1e-4,mean_thickness_m=1e-5,
        four_terminal_resistance_ohm=100.,voltage_probe_spacing_m=.02,conducting_width_m=.01,initial_monomer_mass_kg=2e-6)
    r=analyze_manufacturing(trace,meta,out);m=r['metrics']
    assert m['net_terminal_energy_J']==pytest.approx(.4)
    assert m['net_charge_C']==pytest.approx(.2)
    assert m['imported_specific_energy_J_per_kg']==pytest.approx(4e5)
    assert m['effective_dry_density_kg_per_m3']==pytest.approx(1000)
    assert m['effective_sheet_resistance_ohm_per_square']==pytest.approx(50)
    assert m['effective_conductivity_S_per_m']==pytest.approx(2000)
    assert m['product_to_initial_monomer_mass_ratio']==.5
    assert r['sha256']['trace']==hashlib.sha256(trace.read_bytes()).hexdigest()
    assert '<script>' not in (out/'manufacturing_report.html').read_text()
    assert '&lt;script&gt;' in (out/'manufacturing_report.html').read_text()
    assert any('not chemical conversion' in s for s in r['warnings'])


def test_missing_mass_does_not_create_yield_or_energy_per_mass(tmp_path):
    r=analyze_manufacturing(*inputs(tmp_path,frequency_Hz=10))
    assert 'imported_specific_energy_J_per_kg' not in r['metrics']
    assert any('unresolved' in w for w in r['warnings'])


@pytest.mark.parametrize('extra', [{'source':''},{'units':{}},{'dry_product_mass_kg':0},
    {'mean_thickness_m':-1},{'frequency_Hz':float('nan')},{'four_terminal_resistance_ohm':True}])
def test_metadata_rejections(tmp_path,extra):
    with pytest.raises(ValueError):analyze_manufacturing(*inputs(tmp_path,**extra))


def test_manufacturing_command_line(tmp_path):
    import subprocess,sys
    trace,meta,out=inputs(tmp_path)
    result=subprocess.run([sys.executable,'-m','pedot_sim.cli','analyze-manufacturing',
        '--trace',str(trace),'--metadata',str(meta),'--output-dir',str(out)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads((out/'manufacturing_analysis.json').read_text())['status']=='analyzed_measurements'
