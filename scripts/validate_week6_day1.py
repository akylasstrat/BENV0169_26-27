"""Private structure, fresh-kernel and independent numerical checks for 6.1."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.metadata
import json
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

from build_week6_day1 import ROOT, STEM, FIGURES, TASKS, make_notebook

OUTPUT = ROOT / 'teaching/week6_day1_outputs'

CHECKS = r'''
import importlib.metadata as _metadata
import hashlib as _hashlib
import sys as _sys

assert len(conditions) == 96 and len(boundary_index) == 97
assert boundary_index[-1] == interval_index[0]+pd.Timedelta(hours=48)
assert boundary_index[0] == pd.Timestamp('2025-02-03')
assert boundary_index[-1] == pd.Timestamp('2025-02-05')
assert np.all(operating_inputs.interval_end.to_numpy() == (interval_index+pd.Timedelta(minutes=30)).to_numpy())
np.testing.assert_array_equal(heat_schedule,
    np.where((interval_index.hour>=5)&(interval_index.hour<8),6.,
             np.where((interval_index.hour>=17)&(interval_index.hour<22),4.,0.)))
assert (q_th,E_el,E_th) == (6.,1.,3.)
assert abs(example_next_temperature-20.15277777777778)<1e-12
assert one_r_one_c_step(20,5,0,5.4,4,.5)<20
assert one_r_one_c_step(20,5,5,5.4,4,.5)>one_r_one_c_step(20,5,4,5.4,4,.5)
assert abs(one_r_one_c_step(20,5,15/5.4,5.4,4,.5)-20)<1e-12

# Independent dimensional calculation in W, seconds and J/K.
_T,_To,_q,_R,_C,_dt = 19.3,7.8,3.4,4.8,3.2,.25
_watts = ((_To-_T)/(_R/1000)+1000*_q)
_joule_result = _T+(_dt*3600)*_watts/(_C*3600000)
np.testing.assert_allclose(one_r_one_c_step(_T,_To,_q,_R,_C,_dt),_joule_result,rtol=0,atol=1e-13)

# Distinctive final input must affect the terminal state, not be discarded.
_idx = pd.date_range('2026-01-01',periods=3,freq='30min')
_out = pd.Series(5.,index=_idx)
_zero = pd.Series(0.,index=_idx)
_last = pd.Series([0.,0.,6.],index=_idx)
_base=simulate_one_r_one_c(_out,_zero,20.,5.4,4.,.5)
_changed=simulate_one_r_one_c(_out,_last,20.,5.4,4.,.5)
assert len(_changed)==4 and _changed.index[-1]==_idx[0]+pd.Timedelta(hours=1.5)
np.testing.assert_array_equal(_changed.iloc[:-1],_base.iloc[:-1])
np.testing.assert_allclose(_changed.iloc[-1]-_base.iloc[-1],.75,rtol=0,atol=1e-12)
for _series in [temperature_no_heat,temperature_scheduled,temperature_compact]:
    assert _series.index.equals(boundary_index) and len(_series)==97 and _series.iloc[0]==20
assert (temperature_scheduled>=temperature_no_heat-1e-12).all()

# Closed-form Euler recurrence, including near equilibrium and an unstable step.
for _dt in [.5,4.,24.,48.]:
    _n=int(96/_dt)
    _times=pd.date_range('2026-01-01',periods=_n,freq=pd.Timedelta(hours=_dt))
    for _initial in [18.,32.+1e-8]:
        _response=simulate_one_r_one_c(pd.Series(5.,index=_times),pd.Series(5.,index=_times),_initial,5.4,4.,_dt)
        _oracle=32.+(_initial-32.)*(1-_dt/21.6)**np.arange(_n+1)
        np.testing.assert_allclose(_response,_oracle,rtol=0,atol=5e-13)
_rrows=parameter_table.loc[parameter_table.sweep.eq('R')]
_crows=parameter_table.loc[parameter_table.sweep.eq('C')]
np.testing.assert_allclose(_rrows.T_eq_c,[22.5,32,45])
np.testing.assert_allclose(_rrows.tau_hours,[14,21.6,32])
np.testing.assert_allclose(_crows.T_eq_c,[32,32,32])
np.testing.assert_allclose(_crows.tau_hours,[10.8,21.6,43.2])
assert all(len(s)==193 for s in [*resistance_responses.values(),*capacitance_responses.values()])
for _row in euler_rows:
    _dt=_row['dt_hours']; _frame=euler_results[_dt]
    _hours=_frame.elapsed_hours.to_numpy()
    assert len(_frame)==int(96/_dt)+1 and _hours[-1]==96
    np.testing.assert_allclose(_frame.exact_c,32+(18-32)*np.exp(-_hours/21.6),rtol=0,atol=1e-13)
    np.testing.assert_array_equal(_frame.error_c,_frame.euler_c-_frame.exact_c)
    np.testing.assert_allclose(_row['a'],1-_dt/21.6)
    np.testing.assert_allclose(_row['max_abs_error_c'],abs(_frame.error_c).max())
assert -1 < float(euler_table.set_index('dt_hours').loc[24,'a']) < 0
assert 1-48/21.6 < -1
assert euler_table.max_abs_error_c.is_monotonic_increasing

# Re-run supplied API independently to check recording, timing and common inputs.
_oracle_plant=TwoStateReferencePlant(parameters=parameters,step_hours=.5,
    measurement_noise_std_c=0,process_noise_std_c=0,random_state=12)
_oracle_plant.reset(20,20)
assert reference_states.index.equals(boundary_index)
assert reference_inputs.index.equals(interval_index)
assert reference_states.iloc[0].air_true_c==reference_states.iloc[0].fabric_c==20
for _k,(_time,_row) in enumerate(conditions.iterrows()):
    _mod=.45 if (_time.hour<8 or _time.hour>=17) else 0.
    _result=_oracle_plant.step(_mod,_row.outdoor_temperature_c,_row.solar_irradiance_w_m2,_row.internal_gains_kw)
    _record=reference_states.iloc[_k+1]
    assert _record.name==_time+pd.Timedelta(minutes=30)
    np.testing.assert_allclose([_record.air_true_c,_record.fabric_c],
        [_result['indoor_temperature_true_c'],_result['fabric_temperature_c']],rtol=0,atol=1e-13)
    np.testing.assert_allclose(reference_inputs.iloc[_k].q_th_kw,_result['delivered_heat_kw'],rtol=0,atol=1e-13)
np.testing.assert_array_equal(reference_states.air_measured_c,reference_states.air_true_c)
np.testing.assert_array_equal(reference_inputs.T_out_c,conditions.outdoor_temperature_c)
np.testing.assert_allclose(reference_inputs.P_el_kw*reference_inputs.COP,reference_inputs.q_th_kw,rtol=0,atol=1e-13)
_matched=simulate_one_r_one_c(reference_inputs.T_out_c,reference_inputs.q_th_kw,20.,5.4,4.,.5)
np.testing.assert_array_equal(temperature_compact,_matched)
assert (reference_inputs.solar_gain_kw>0).any()
assert parameters.gains_to_air_fraction==.45 and parameters.heating_to_air_fraction==.25
assert parameters.resistance_outdoor_fabric_k_per_kw==6 and parameters.resistance_air_fabric_k_per_kw==.8

# Fresh instances, not reset(), reproduce optional sensor noise.
_noise_plant=TwoStateReferencePlant(parameters=parameters,step_hours=.5,
    measurement_noise_std_c=.05,process_noise_std_c=0.,random_state=12)
_noise=[_noise_plant.reset(20,20)]
for _time,_row in conditions.iterrows():
    _result=_noise_plant.step(reference_inputs.loc[_time,'modulation'],_row.outdoor_temperature_c,_row.solar_irradiance_w_m2,_row.internal_gains_kw)
    _noise.append(_result['indoor_temperature_measured_c'])
np.testing.assert_array_equal(optional_noise.measured_air_c,_noise)
np.testing.assert_array_equal(optional_noise.true_air_c,reference_states.air_true_c)

# Every unfinished task fails at its own helpful exception, including functions.
_student=json.loads((ROOT/'tutorial_6_1_control_oriented_building_model.ipynb').read_text(encoding='utf8'))
for _cell in _student['cells']:
    _tags=_cell.get('metadata',{}).get('tags',[])
    if 'task' not in _tags: continue
    _letter=_tags[-1]; _scope=dict(globals())
    try:
        exec(''.join(_cell['source']),_scope)
        if _letter=='B':_scope['one_r_one_c_step'](20,5,4,5.4,4,.5)
        elif _letter=='C':_scope['simulate_one_r_one_c'](_out,_zero,20,5.4,4,.5)
    except NotImplementedError as _error:
        assert 'Task '+_letter in str(_error)
    else: raise AssertionError('Unfinished task did not fail: '+_letter)

validation_payload={
    'python':_sys.version,'executable':_sys.executable,
    'packages':{name:_metadata.version(name) for name in ['numpy','pandas','matplotlib','scipy','scikit-learn','nbformat','nbclient','ipykernel']},
    'task_A':{'q_th_kw':q_th,'E_el_kWh':E_el,'E_th_kWh':E_th},
    'single_step_c':example_next_temperature,'euler':euler_rows,
    'parameter_sweep':parameter_table.to_dict(orient='records'),
    'comparison':comparison_summary.to_dict(orient='records'),
    'schedule_terminal_c':rollout_states.iloc[-1].to_dict(),
    'core_prediction_sha256':_hashlib.sha256(rollout_states.to_numpy().tobytes()+
        comparison_states.to_numpy().tobytes()).hexdigest(),
    'checks':'PASS: units, sign, equilibrium, closed-form Euler, final input, boundary timing, matched plant inputs, noise reproducibility, all six TODO failures'}
print(validation_payload['checks'])
'''


def static_checks():
    pair = [nbformat.read(ROOT/(STEM+s+'.ipynb'),4) for s in ['', '_solution']]
    for nb, solution in zip(pair,[False,True]):
        nbformat.validate(nb)
        expected = make_notebook(solution)
        assert [(c.id,c.source) for c in nb.cells] == [(c.id,c.source) for c in expected.cells]
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,cell.id,'exec')
    student, solution = pair
    assert len(student.cells)==len(solution.cells)
    letters=[]
    for a,b in zip(student.cells,solution.cells):
        assert (a.id,a.cell_type,a.metadata.get('tags'))==(b.id,b.cell_type,b.metadata.get('tags'))
        if a.cell_type=='code':assert a.execution_count is None and not a.outputs
        if 'task' in a.metadata.get('tags',[]):
            letter=a.metadata.tags[-1]; letters.append(letter)
            assert 'TODO '+letter in a.source and 'NotImplementedError' in a.source
            assert 'NotImplementedError' not in b.source
        else:assert a.source==b.source
    assert letters==list('ABCDEF')
    assert all('iloc[:-1]' not in c.source for c in student.cells)
    # Only mathematical expressions and optional supplied reference remain outside tasks;
    # no hidden controller algorithm is imported from course_utils.
    imports=[n for c in student.cells if c.cell_type=='code' for n in ast.walk(ast.parse(c.source))
             if isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('course_utils')]
    assert len(imports)==1 and imports[0].module=='course_utils.reference_building'
    assert {a.name for a in imports[0].names}=={'TwoStateReferencePlant','ReferenceBuildingParameters'}
    return pair


def execute(nb, kernel_root, result_path):
    executed=copy.deepcopy(nb)
    executed.cells.append(nbformat.v4.new_code_cell(CHECKS+'\nPath('+repr(str(result_path))+').write_text(json.dumps(validation_payload,indent=2),encoding="utf8")'))
    manager=KernelManager(kernel_name='week6-day1-validation',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client=NotebookClient(executed,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}})
    try:client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    executed.cells.pop()
    return executed


def check_files():
    import pandas as pd
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week6_day1_figure_manifest.csv').fillna('')
    assert manifest.id.tolist()==[row[0] for row in FIGURES] and manifest.id.is_unique
    tasks=pd.read_csv(ROOT/'teaching/week6_day1_task_manifest.csv')
    assert tasks.task_id.tolist()==list('ABCDEF') and tasks.minutes_including_debrief.sum()==62
    assert tasks.slide.tolist()==[item[2] for item in TASKS]
    assert tasks.cell_id.tolist()==['w61-task-'+letter.lower() for letter in 'ABCDEF']
    for row in manifest.itertuples():
        for name in [row.png,row.svg]:assert (ROOT/name).is_file()
        with Image.open(ROOT/row.png) as im:
            assert im.width>1200 and abs(im.info['dpi'][0]-220)<.1
        svg=(ROOT/row.svg).read_text(encoding='utf8')
        assert '<text' in svg
        assert row.slide==next(spec[2] for spec in FIGURES if spec[0]==row.id)
        files=row.source_files.split(';') if row.source_files else []
        assert ';'.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files)==row.source_sha256
    frozen=json.loads((OUTPUT/'protected_hashes.json').read_text(encoding='utf8'))
    for name,digest in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return len(frozen), manifest


def main():
    student, solution = static_checks()
    temporary=Path(tempfile.mkdtemp(prefix='BENV0169_week6_day1_'))
    kernel_root=temporary/'kernels'; spec=kernel_root/'week6-day1-validation'; spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 6.1 validation','language':'python'}),encoding='utf8')
    executed=execute(solution,kernel_root,temporary/'solution_checks.json')
    nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Fresh-kernel solution and independent numerical checks passed.',flush=True)
    completed=copy.deepcopy(student)
    for a,b in zip(completed.cells,solution.cells):
        if 'task' in a.metadata.get('tags',[]):a.source=b.source
    executed_student=execute(completed,kernel_root,temporary/'completed_checks.json')
    nbformat.write(executed_student,temporary/'completed_student_PRIVATE.ipynb')
    first=json.loads((temporary/'solution_checks.json').read_text(encoding='utf8'))
    second=json.loads((temporary/'completed_checks.json').read_text(encoding='utf8'))
    assert first==second
    count, manifest=check_files()
    static_checks()
    first.update({'status':'PASS','fresh_kernel_runs':2,'private_evidence_directory':str(temporary),
        'unchanged_protected_files':count,'figure_ids':manifest.id.tolist(),
        'copied_plant_sha256':hashlib.sha256((ROOT/'course_utils/reference_building.py').read_bytes()).hexdigest(),
        'command':f'{sys.executable} -B scripts/validate_week6_day1.py'})
    (OUTPUT/'validation_results.json').write_text(json.dumps(first,indent=2),encoding='utf8')
    print(f'Completed student copy passed; {count} protected hashes unchanged; all ten PNG/SVG pairs verified.',flush=True)
    print(json.dumps({'euler':first['euler'],'comparison':first['comparison']},indent=2))


if __name__=='__main__':main()
