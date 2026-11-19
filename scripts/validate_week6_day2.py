"""Private clean-kernel, alignment, holdout and seasonal-model checks for 6.2."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import uuid
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from build_week6_day2 import ROOT, STEM, FIGURES, TASKS, SLIDE_MINUTES, TEACHING_BLOCKS, make_notebook

OUTPUT=ROOT/'teaching/week6_day2_outputs'
CORE_CHECKS=r'''
assert len(data)==1344 and data.index[0]==pd.Timestamp('2025-01-06') and data.index[-1]==pd.Timestamp('2025-02-02 23:30')
assert [len(fit_observations),len(validation_observations)]==[1008,336]
assert [len(fit_transitions),len(validation_transitions)]==[1007,335]
for _frame in [fit_transitions,validation_transitions]:
    assert (_frame.target_time-_frame.predictor_time).eq(pd.Timedelta(minutes=30)).all()
    assert np.isfinite(_frame[required+['next_temperature_c']]).all().all()
    np.testing.assert_array_equal(_frame.next_temperature_c,data.loc[_frame.target_time,'indoor_temperature_c'])
assert fit_transitions.target_time.max()<split_time
assert fit_transitions.predictor_time.max()==pd.Timestamp('2025-01-26 23:00')
assert validation_transitions.predictor_time.min()==split_time
assert actual.index[0]==split_time+pd.Timedelta(minutes=30)
assert actual.index.equals(one_step.index) and actual.index.equals(rollout.index) and len(actual)==335
assert split_time not in actual.index
assert (score_table.N==335).all()
assert bounded_fit.success and (bounded_fit.x>=LOWER_BOUNDS).all() and (bounded_fit.x<=UPPER_BOUNDS).all()
np.testing.assert_allclose(estimated_C,.5/c);np.testing.assert_allclose(estimated_R,c/b)
np.testing.assert_allclose(estimated_tau,.5/b);np.testing.assert_allclose(a,1-b)
if ((ols_coefficients>=LOWER_BOUNDS)&(ols_coefficients<=UPPER_BOUNDS)).all():
    np.testing.assert_allclose(bounded_fit.x,ols_coefficients,rtol=0,atol=1e-12)

# Independent known one-state fixture with informative, noiseless inputs.
_rng=np.random.default_rng(731)
_outside=_rng.uniform(0,12,200);_heat=_rng.uniform(0,10,200)
_states=[20.]
for _out,_q in zip(_outside,_heat):_states.append(.97*_states[-1]+.03*_out+.12*_q)
_X=np.column_stack([_outside-np.array(_states[:-1]),_heat])
_y=np.diff(_states)
assert np.linalg.matrix_rank(_X)==2
_fit=lsq_linear(_X,_y,bounds=(LOWER_BOUNDS,UPPER_BOUNDS))
np.testing.assert_allclose(_fit.x,[.03,.12],rtol=0,atol=1e-12)
_invalid=lsq_linear(np.eye(2),[-.2,-.1],bounds=(LOWER_BOUNDS,UPPER_BOUNDS))
assert _invalid.success and (_invalid.active_mask==-1).all()
np.testing.assert_allclose(_invalid.x,LOWER_BOUNDS,rtol=0,atol=1e-7)

# Execute actual task source against altered measurements without changing inputs.
_notebook=json.loads((ROOT/'tutorial_6_2_rc_model_identification_solution.ipynb').read_text(encoding='utf8'))
_sources={cell['id']:''.join(cell['source']) for cell in _notebook['cells']}
def _prediction_fixture(observations):
    scope=dict(globals());scope['validation_observations']=observations.copy(deep=True)
    exec(_sources['w62-task-a'],scope);exec(_sources['w62-task-d'],scope)
    return scope
_original=_prediction_fixture(validation_observations)
_modified=validation_observations.copy(deep=True)
_modified.iloc[20,_modified.columns.get_loc('indoor_temperature_c')]+=2
_perturbed=_prediction_fixture(_modified)
np.testing.assert_array_equal(_perturbed['rollout'],_original['rollout'])
np.testing.assert_allclose(_perturbed['one_step'].iloc[20]-_original['one_step'].iloc[20],2*(1-b))
_initial=validation_observations.copy(deep=True)
_initial.iloc[0,_initial.columns.get_loc('indoor_temperature_c')]+=1
_initial_result=_prediction_fixture(_initial)
np.testing.assert_allclose(_initial_result['rollout']-_original['rollout'],(1-b)**np.arange(1,336),atol=1e-12)
np.testing.assert_allclose(_initial_result['one_step'].iloc[0]-_original['one_step'].iloc[0],1-b)

# Missing or non-finite transitions are excluded rather than bridged.
_irregular=validation_observations.drop(validation_observations.index[12])
_scope=dict(globals());_scope['validation_observations']=_irregular
exec(_sources['w62-task-a'],_scope)
assert len(_scope['validation_transitions'])==333
assert (_scope['validation_transitions'].target_time-_scope['validation_transitions'].predictor_time).eq(pd.Timedelta(minutes=30)).all()
_nonfinite=validation_observations.copy();_nonfinite.iloc[10,_nonfinite.columns.get_loc('delivered_heat_kw')]=np.nan
_scope=dict(globals());_scope['validation_observations']=_nonfinite;exec(_sources['w62-task-a'],_scope)
assert len(_scope['validation_transitions'])==334

# Hand metric oracle and consistent actual-minus-predicted sign.
_hand=temperature_metrics(pd.Series([-1.,0.,3.]),pd.Series([0.,0.,0.]))
np.testing.assert_allclose([_hand['MAE_K'],_hand['RMSE_K'],_hand['bias_K']],[4/3,np.sqrt(10/3),2/3])
try:temperature_metrics(pd.Series([1.],index=[0]),pd.Series([1.],index=[1]))
except ValueError:pass
else:raise AssertionError('Metric alignment not enforced')
for _name,_prediction in prediction_cases.items():
    _row=score_table.set_index('method').loc[_name]
    _errors=actual.to_numpy()-_prediction.to_numpy()
    np.testing.assert_allclose(_row[['MAE_K','RMSE_K','bias_K']].to_numpy(dtype=float),
        [sum(abs(_errors))/335,np.sqrt(sum(_errors**2)/335),sum(_errors)/335],rtol=0,atol=1e-12)

def _check_interval_inputs(records):
    assert (records.target_time-records.predictor_time).eq(pd.Timedelta(minutes=30)).all()
    np.testing.assert_array_equal(records.delivered_heat_kw,data.loc[records.predictor_time,'delivered_heat_kw'])
    np.testing.assert_array_equal(records.outdoor_temperature_c,data.loc[records.predictor_time,'outdoor_temperature_c'])
_check_interval_inputs(residual_records)
assert [t.get_text() for t in ax2.get_xticklabels()]==[str(h) for h in range(0,24,3)]
assert residual_records.groupby('target_hour').size().tolist()==[13]+[14]*23
_shifted=residual_records.copy();_shifted['predictor_time']+=pd.Timedelta(minutes=30)
try:_check_interval_inputs(_shifted)
except AssertionError:pass
else:raise AssertionError('Shifted interval inputs were not caught')
_permuted=residual_records.copy();_permuted['predictor_time']=_permuted.predictor_time.to_numpy()[::-1]
try:_check_interval_inputs(_permuted)
except AssertionError:pass
else:raise AssertionError('Permuted interval inputs were not caught')


'''

HOUSE_CHECKS=r'''
_notebook=json.loads((ROOT/'tutorial_6_2_rc_model_identification_solution.ipynb').read_text(encoding='utf8'))
_sources={cell['id']:''.join(cell['source']) for cell in _notebook['cells']}

# Independently check the hand feature row and solar-unit conversion.
_index=pd.date_range('2020-01-01',periods=5,freq='10min')
_fixture=pd.DataFrame({'T_Average (degC)':[20.,20.1,20.2,20.3,20.4],
    'T_External (degC)':10.,'P_tot (W)':2000.,'Solar (W/m2)':500.},index=_index)
_transitions=make_house_transitions(_fixture)
np.testing.assert_allclose(_transitions[house_features].iloc[0],[-10,2,.5])
np.testing.assert_allclose(_transitions.dT.iloc[0],.1)
assert [len(november_observations),len(june_fit_observations),len(june_test_observations)]==[720,432,288]
assert [len(november_fit),len(june_fit),len(june_test)]==[719,431,287]
assert june_fit.target_time.max()<june_split
assert june_test.index.min()>=june_split and june_test.target_time.min()>june_split
assert len(june_fit.index.intersection(june_test.index))==0
assert house_features==['T_diff','P_kW','S_kW_m2']
for _frame,_raw in [(november_fit,november_observations),(june_fit,june_fit_observations),(june_test,june_test_observations)]:
    assert ((_frame.target_time-_frame.index)==pd.Timedelta(minutes=10)).all()
    np.testing.assert_array_equal(_frame.T_next,_raw.loc[_frame.target_time,'T_Average (degC)'])
    np.testing.assert_array_equal(_frame.T_diff,_raw.loc[_frame.index,'T_External (degC)']-_frame['T'])
    np.testing.assert_array_equal(_frame.P_kW,_raw.loc[_frame.index,'P_tot (W)']/1000)
    np.testing.assert_array_equal(_frame.S_kW_m2,_raw.loc[_frame.index,'Solar (W/m2)']/1000)
    assert np.isfinite(_frame[house_features+['T','T_next','dT']]).all().all()

for _name,_frame,_model in [('November',november_fit,november_model),('June',june_fit,june_model)]:
    assert _model.success and (_model.x>=house_lower).all() and (_model.x<=house_upper).all()
    _independent=lsq_linear(_frame[house_features].to_numpy(),_frame.dT.to_numpy(),bounds=(house_lower,house_upper))
    np.testing.assert_allclose(_model.x,_independent.x,rtol=0,atol=1e-12)
    _b,_c,_d=_model.x
    _row=house_parameters.set_index('fit_period').loc[_name]
    np.testing.assert_allclose(_row[['R_eff_K_per_kW','C_eff_kWh_per_K','tau_hours','g_eff_m2']].to_numpy(dtype=float),
                               [_c/_b,house_dt/_c,house_dt/_b,_d/_c])

# Independent recovery of known solar-inclusive coefficients.
_rng=np.random.default_rng(62)
_X=np.column_stack([_rng.uniform(-15,5,100),_rng.uniform(0,6,100),_rng.uniform(0,1,100)])
_known=np.array([.01,.05,.1])
np.testing.assert_allclose(lsq_linear(_X,_X@_known,bounds=(house_lower,house_upper)).x,_known,atol=1e-12)

for _name,_theta in [('November transfer',november_coefficients),('June refit',june_coefficients)]:
    _expected=june_test['T'].to_numpy()+june_test[house_features].to_numpy()@_theta
    np.testing.assert_allclose(june_predictions[_name],_expected,rtol=0,atol=1e-12)
assert june_scores.N.eq(len(june_test)).all()
for _name,_prediction in june_predictions.items():
    _errors=june_test.T_next.to_numpy()-_prediction.to_numpy()
    _row=june_scores.set_index('model').loc[_name]
    np.testing.assert_allclose(_row[['MAE_K','RMSE_K','bias_K']].to_numpy(dtype=float),
        [np.mean(abs(_errors)),np.sqrt(np.mean(_errors**2)),np.mean(_errors)],atol=1e-12)

# Alter held-out temperatures and rerun the actual split and fit cells.
_scope=dict(globals())
_scope.update(house_raw={key:frame.copy(deep=True) for key,frame in house_raw.items()},
              display=lambda *args:None,export_table=lambda *args,**kwargs:None)
_raw=_scope['house_raw']['June']
_raw.loc[_raw.index>=june_split,'T_Average (degC)']+=5
exec(_sources['w62-house-split'],_scope);exec(_sources['w62-task-r2'],_scope)
np.testing.assert_array_equal(_scope['november_coefficients'],november_coefficients)
np.testing.assert_array_equal(_scope['june_coefficients'],june_coefficients)

validation_payload={'house_counts':house_counts.assign(first_predictor=house_counts.first_predictor.astype(str),
    last_target=house_counts.last_target.astype(str)).to_dict('records'),
    'coefficients':coefficient_table.to_dict('index'),'effective_parameters':house_parameters.to_dict('records'),
    'june_test_scores':june_scores.to_dict('records'),
    'checks':'PASS: interpolated alignment, solar units, coefficient recovery, separate fits, held-out independence and common-target scoring'}
'''

CHECKS=CORE_CHECKS+HOUSE_CHECKS


def static_checks():
    pair=[nbformat.read(ROOT/(STEM+suffix+'.ipynb'),4) for suffix in ['', '_solution']]
    for notebook,solved in zip(pair,[False,True]):
        nbformat.validate(notebook)
        assert [(c.id,c.source) for c in notebook.cells]==[(c.id,c.source) for c in make_notebook(solved).cells]
        for c in notebook.cells:
            if c.cell_type=='code':compile(c.source,c.id,'exec')
            else:assert chr(92)*2 not in c.source,c.id
    task_ids=[]
    for student,solution in zip(*[n.cells for n in pair]):
        assert (student.id,student.cell_type,student.metadata.get('tags'))==(solution.id,solution.cell_type,solution.metadata.get('tags'))
        if student.cell_type=='code':assert not student.outputs and student.execution_count is None
        if 'task' in student.metadata.get('tags',[]):
            task_ids.append(student.metadata.tags[-1])
            assert 'NotImplementedError' in student.source and 'NotImplementedError' not in solution.source
        else:assert student.source==solution.source
    assert task_ids==[row[0] for row in TASKS]==['A','B','C','D','E','R1','R2','R3']
    text='\n'.join(c.source for c in pair[0].cells)
    for obsolete in ['Task F','horizon_records','predict_house_segments','June post-hoc','Optional real-data']:
        assert obsolete not in text,obsolete
    assert sum(SLIDE_MINUTES)==200
    assert [sum(SLIDE_MINUTES[first-1:last]) for first,last in TEACHING_BLOCKS]==[50]*4
    assert all(SLIDE_MINUTES[t[2]-1]==t[3] for t in TASKS)
    return pair


def execute(notebook,kernel_root,path,checks,export=False):
    executed=copy.deepcopy(notebook)
    setup=next(c for c in executed.cells if c.id=='w62-setup')
    setup.source+='\nEXPORT_ARTEFACTS='+repr(export)
    executed.cells.append(nbformat.v4.new_code_cell(checks+'\nPath('+repr(str(path))+').write_text(json.dumps(validation_payload,indent=2),encoding="utf8")'))
    manager=KernelManager(kernel_name='week6-day2-validation',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client=NotebookClient(executed,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}})
    try:client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    executed.cells.pop()
    setup.source=setup.source.rsplit('\nEXPORT_ARTEFACTS=',1)[0]
    return executed


def check_outputs():
    import pandas as pd
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week6_day2_figure_manifest.csv').fillna('')
    assert manifest.id.tolist()==sorted(row[0] for row in FIGURES)
    for row in manifest.itertuples():
        assert str(row.slide)==next(spec[2] for spec in FIGURES if spec[0]==row.id)
        for name in [row.png,row.svg]:assert (ROOT/name).is_file()
        with Image.open(ROOT/row.png) as im:assert im.width>1200 and abs(im.info['dpi'][0]-220)<.1
        assert '<text' in (ROOT/row.svg).read_text(encoding='utf8')
        files=row.source_files.split(';') if row.source_files else []
        assert ';'.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files)==row.source_sha256
    tasks=pd.read_csv(ROOT/'teaching/week6_day2_task_manifest.csv')
    assert tasks.task_id.tolist()==[row[0] for row in TASKS]
    assert tasks.minutes_including_debrief.sum()==92


def main():
    student,solution=static_checks()
    (ROOT/'tmp').mkdir(exist_ok=True)
    temporary=ROOT/'tmp'/('week6_day2_validation_'+uuid.uuid4().hex)
    temporary.mkdir()
    kernel_root=temporary/'kernels';spec=kernel_root/'week6-day2-validation';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 6.2 validation','language':'python'}),encoding='utf8')
    executed=execute(solution,kernel_root,temporary/'solution_checks.json',CHECKS,export=True)
    nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Fresh-kernel solution and independent numerical checks passed.',flush=True)
    completed=copy.deepcopy(student)
    for scaffold,answer in zip(completed.cells,solution.cells):
        if 'task' in scaffold.metadata.get('tags',[]):scaffold.source=answer.source
    private=execute(completed,kernel_root,temporary/'completed_checks.json',CHECKS)
    nbformat.write(private,temporary/'completed_student_PRIVATE.ipynb')
    result=json.loads((temporary/'solution_checks.json').read_text(encoding='utf8'))
    assert result==json.loads((temporary/'completed_checks.json').read_text(encoding='utf8'))
    print('Privately completed student notebook matches.',flush=True)
    independent=copy.deepcopy(completed)
    wanted={'w62-setup','w62-helpers','w62-metrics'}
    first=next(i for i,c in enumerate(independent.cells) if c.id=='w62-house-data')
    independent.cells=[c for i,c in enumerate(independent.cells) if c.id in wanted or i>=first]
    execute(independent,kernel_root,temporary/'independent_house_checks.json',HOUSE_CHECKS)
    assert result==json.loads((temporary/'independent_house_checks.json').read_text(encoding='utf8'))
    check_outputs();static_checks()
    result.update({'status':'PASS','fresh_kernel_runs':3,'private_evidence_directory':str(temporary),
        'python':sys.version,'executable':sys.executable})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'validation_results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
