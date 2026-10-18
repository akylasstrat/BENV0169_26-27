"""Private clean-kernel, alignment, holdout and measured-mask checks for 6.2."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from build_week6_day2 import ROOT, STEM, FIGURES, TASKS, SLIDE_MINUTES, TEACHING_BLOCKS, make_notebook

OUTPUT=ROOT/'teaching/week6_day2_outputs'
CHECKS=r'''
import importlib.metadata as _metadata
import hashlib as _hashlib
import sys as _sys

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

# Complete, identical origin sets and independent lead/window reconciliation.
assert origins==list(pd.date_range('2025-01-27',periods=6,freq='D'))
assert len(horizon_records)==288 and horizon_records.groupby('origin').size().eq(48).all()
assert horizon_records.groupby('lead_steps').size().eq(6).all()
assert (horizon_records.target_time-horizon_records.origin).eq(pd.to_timedelta(horizon_records.lead_hours,unit='h')).all()
assert (horizon_records.target_time-horizon_records.predictor_time).eq(pd.Timedelta(minutes=30)).all()
np.testing.assert_array_equal(horizon_records.actual_c,data.loc[horizon_records.target_time,'indoor_temperature_c'])
_error_matrix=np.stack([group.sort_values('lead_steps').error_c.to_numpy() for _,group in horizon_records.groupby('origin')])
np.testing.assert_allclose(lead_rmse,np.sqrt(np.mean(_error_matrix**2,axis=0)),rtol=0,atol=1e-12)
for _h in [6,12,24]:
    _window=window_scores.loc[window_scores.horizon_hours.eq(_h)]
    assert _window.origin.tolist()==origins and _window.N.eq(int(_h/dt)).all()
    np.testing.assert_allclose(np.mean(_window.RMSE_K**2),np.mean(lead_rmse.loc[:_h]**2),rtol=0,atol=1e-12)

# Matched benchmark: same six-origin grid, independent measured-state formula.
_predictors=data.loc[horizon_records.predictor_time]
_expected_one_step=(1-b)*_predictors.indoor_temperature_c.to_numpy()+b*_predictors.outdoor_temperature_c.to_numpy()+c*_predictors.delivered_heat_kw.to_numpy()
np.testing.assert_allclose(horizon_records.one_step_prediction_c,_expected_one_step,rtol=0,atol=1e-12)
np.testing.assert_allclose(horizon_records.one_step_error_c,horizon_records.actual_c-_expected_one_step,rtol=0,atol=1e-12)
assert np.isfinite(horizon_records[['prediction_c','error_c','one_step_prediction_c','one_step_error_c']]).all().all()
_first=horizon_records.loc[horizon_records.lead_steps.eq(1)]
np.testing.assert_allclose(_first.prediction_c,_first.one_step_prediction_c,rtol=0,atol=1e-12)
_one_step_matrix=np.stack([g.sort_values('lead_steps').one_step_error_c.to_numpy() for _,g in horizon_records.groupby('origin')])
np.testing.assert_allclose(one_step_lead_rmse,np.sqrt(np.mean(_one_step_matrix**2,axis=0)),rtol=0,atol=1e-12)
assert benchmark_lead_summary.N_origins.eq(6).all()
assert first_origin_windows.N_targets.tolist()==[12,24,48]
assert first_origin_windows.start.eq(pd.Timestamp('2025-01-27')).all()
assert first_origin_windows.end.tolist()==[pd.Timestamp('2025-01-27 06:00'),pd.Timestamp('2025-01-27 12:00'),pd.Timestamp('2025-01-28')]
_scope=dict(globals());_scope.update(validation_observations=_modified,EXPORT_ARTEFACTS=False,
    display=lambda *args:None,export_table=lambda *args,**kwargs:None)
exec(_sources['w62-horizon-experiment'],_scope)
np.testing.assert_array_equal(_scope['horizon_records'].prediction_c,horizon_records.prediction_c)
_affected=(horizon_records.predictor_time==_modified.index[20])
np.testing.assert_allclose(_scope['horizon_records'].one_step_prediction_c-horizon_records.one_step_prediction_c,2*(1-b)*_affected,atol=1e-12)

# Headline house targets and fit inputs are original measurements, not imputed truth.
assert len(house_raw['November'])==len(house_raw['June'])==720
assert [int(house_predictions[name].common_score_mask.sum()) for name in ['November','June']]==[698,694]
for _name,_raw in house_raw.items():
    _table=house_predictions[_name]
    _mask=house_original_masks[_name]
    _expected=_mask.all(axis=1)&_mask[HOUSE_COLUMNS[0]].shift(-1,fill_value=False)
    np.testing.assert_array_equal(_table.original_eligible,_expected.iloc[:-1])
    np.testing.assert_array_equal(_table.loc[_table.original_eligible,'next_temperature_c'],
        _raw.loc[_table.loc[_table.original_eligible,'target_time'],HOUSE_COLUMNS[0]])
    assert (_table.common_score_mask<=_table.original_eligible).all()
    assert _table.loc[_table.reset_at_predictor,'indoor_temperature_c'].notna().all()
    assert (_table.target_time-_table.predictor_time).eq(pd.Timedelta(minutes=10)).all()
    for _method,_column in [('One-step','one_step_c'),('Segmented rollout','rollout_c')]:
        _use=_table.loc[_table.common_score_mask]
        _errors=_use.next_temperature_c-_use[_column]
        _row=house_scores.loc[(house_scores.period==_name)&(house_scores.method==_method)].iloc[0]
        np.testing.assert_allclose(_row[['MAE_K','RMSE_K','bias_K']].to_numpy(dtype=float),
            [np.mean(abs(_errors)),np.sqrt(np.mean(_errors**2)),np.mean(_errors)],rtol=0,atol=1e-12)
assert len(winter_transitions)==698 and house_fit.success
np.testing.assert_array_equal(house_coefficients,extension_coefficients)
assert np.array_equal(house_coefficients,house_fit.x) # June sensitivity did not replace winter coefficients.
assert gap_sensitivity_scores.N_segments.eq(1).all()
assert provenance['common_model']['fit_provenance'].startswith('all 28')

# A missing state observation does not reset a continuing rollout; missing forcing does.
_fixture=house_tables['November'].head(8).copy()
_baseline=predict_house_segments(_fixture,house_coefficients)
_temperature_gap=_fixture.copy();_temperature_gap.iloc[3,_temperature_gap.columns.get_loc('indoor_temperature_c')]=np.nan
_tg=predict_house_segments(_temperature_gap,house_coefficients)
np.testing.assert_array_equal(_tg.rollout_c,_baseline.rollout_c)
assert _tg.reset_at_predictor.sum()==1
_forcing_gap=_fixture.copy();_forcing_gap.iloc[3,_forcing_gap.columns.get_loc('outdoor_temperature_c')]=np.nan
_fg=predict_house_segments(_forcing_gap,house_coefficients)
assert np.isnan(_fg.rollout_c.iloc[3]) and _fg.reset_at_predictor.iloc[4]
assert _fg.reset_at_predictor.sum()==2

# E1 must not fill endpoints or partially fill a long run.
_idx=pd.date_range('2026-01-01',periods=11,freq='10min')
_v=[np.nan,1.,np.nan,np.nan,4.,np.nan,np.nan,np.nan,np.nan,9.,np.nan]
_raw_fixture=pd.DataFrame({col:_v for col in HOUSE_COLUMNS},index=_idx)
_scope=dict(globals());_scope['house_raw']={'fixture':_raw_fixture.copy()}
exec(_sources['w62-task-e1'],_scope)
_filled=_scope['short_filled']['fixture']
assert _filled.iloc[[0,5,6,7,8,10]].isna().all().all()
np.testing.assert_allclose(_filled.iloc[2:4].to_numpy(),[[2]*4,[3]*4])
pd.testing.assert_frame_equal(_scope['house_raw']['fixture'],_raw_fixture)
for _name,_raw in house_raw.items():
    _observed=house_original_masks[_name]
    for _col in HOUSE_COLUMNS:
        pd.testing.assert_series_equal(short_filled[_name].loc[_observed[_col],_col],_raw.loc[_observed[_col],_col])
assert sensitivity_windows['Nov first 48h'].target_time.max()<house_raw['November'].index[0]+pd.Timedelta(hours=48)
assert sensitivity_windows['Nov final 48h'].predictor_time.min()==pd.Timestamp('2016-11-14')
assert sensitivity_table.loc[sensitivity_table.window.eq('June post-hoc'),'role'].iloc[0]=='post-hoc in-sample'
assert sensitivity_table.N_fit.tolist()==[698,278,278,694]
assert (sensitivity_table.active_bounds=='[0.0, 0.0, 0.0]').all()

# Every TODO has its own intentional, informative failure.
_student=json.loads((ROOT/'tutorial_6_2_rc_model_identification.ipynb').read_text(encoding='utf8'))
for _cell in _student['cells']:
    _tags=_cell.get('metadata',{}).get('tags',[])
    if not any(tag in _tags for tag in ['task','optional_task']):continue
    try:exec(''.join(_cell['source']),dict(globals()))
    except NotImplementedError as _error:assert _tags[-1] in str(_error)
    else:raise AssertionError('TODO did not fail: '+_tags[-1])

validation_payload={'python':_sys.version,'executable':_sys.executable,
    'packages':{p:_metadata.version(p) for p in ['numpy','pandas','matplotlib','scipy','scikit-learn','nbformat','nbclient','ipykernel']},
    'N_fit':len(fit_transitions),'N_validation':len(actual),'fit_comparison':fit_comparison.to_dict(orient='records'),
    'parameters':parameter_table.to_dict(orient='records'),'validation_scores':score_table.to_dict(orient='records'),
    'window_scores':window_scores.assign(origin=window_scores.origin.astype(str)).to_dict(orient='records'),
    'matched_benchmark':{'N_origins':len(origins),'N_leads':len(one_step_lead_rmse),'N_targets':len(horizon_records),
        'first_lead_one_step_RMSE_K':float(one_step_lead_rmse.iloc[0]),'first_lead_rollout_RMSE_K':float(lead_rmse.iloc[0])},
    'house_model':house_model_record.to_dict(orient='records'),'house_scores':house_scores.to_dict(orient='records'),
    'input_gap_sensitivity':gap_sensitivity_scores.to_dict(orient='records'),
    'house_segments':house_segments.groupby('period').size().to_dict(),
    'prediction_sha256':_hashlib.sha256(one_step.to_numpy().tobytes()+rollout.to_numpy().tobytes()+house_coefficients.tobytes()).hexdigest(),
    'checks':'PASS: within-period timing, deterministic coefficient recovery, active bounds, reset information, metric sign, interval-input negatives, complete horizons, independent matched one-step benchmark, observed masks, explicit segments, limited interpolation and post-hoc separation'}
print(validation_payload['checks'])
'''

DEMO_CHECKS='''
assert len(house_raw['November'])==len(house_raw['June'])==720
assert house_scores.groupby('period').N.first().to_dict()=={'June':694,'November':698}
validation_payload={'house_model':house_model_record.to_dict(orient='records'),'house_scores':house_scores.to_dict(orient='records')}
print('Standalone real-house demonstration passed without synthetic tasks or optional exercises.')
'''


def static_checks():
    pair=[nbformat.read(ROOT/(STEM+suffix+'.ipynb'),4) for suffix in ['', '_solution']]
    for notebook,solved in zip(pair,[False,True]):
        nbformat.validate(notebook)
        expected=make_notebook(solved)
        assert [(c.id,c.source) for c in notebook.cells]==[(c.id,c.source) for c in expected.cells]
        for c in notebook.cells:
            if c.cell_type=='code':compile(c.source,c.id,'exec')
    core=[];optional=[]
    for student,solution in zip(*[n.cells for n in pair]):
        assert (student.id,student.cell_type,student.metadata.get('tags'))==(solution.id,solution.cell_type,solution.metadata.get('tags'))
        if student.cell_type=='code':assert not student.outputs and student.execution_count is None
        tags=student.metadata.get('tags',[])
        if 'task' in tags or 'optional_task' in tags:
            (core if 'task' in tags else optional).append(tags[-1])
            assert 'NotImplementedError' in student.source and 'NotImplementedError' not in solution.source
        else:assert student.source==solution.source
    assert core==list('ABCDEF') and optional==['E1','E2','E3']
    ids=[c.id for c in pair[0].cells]
    assert ids.index('w62-house-transfer-figure')<ids.index('w62-task-e1')
    assert ids.index('w62-core-manifest')<ids.index('w62-task-e1')
    assert all('verified_controller_model' not in c.source for c in pair[0].cells[:ids.index('w62-i8')])
    assert [t[2] for t in TASKS]==[7,13,15,19,21,25]
    assert [t[3] for t in TASKS]==[12,12,8,12,8,12]
    assert [f[2] for f in FIGURES]==['4','5','6;8','14','18','20','22','24','26','28','35 (optional)']
    assert len(SLIDE_MINUTES)==31 and sum(SLIDE_MINUTES)==200
    assert [sum(SLIDE_MINUTES[first-1:last]) for first,last in TEACHING_BLOCKS]==[50]*4
    assert all(SLIDE_MINUTES[t[2]-1]==t[3] for t in TASKS)
    text='\n'.join(c.source for c in pair[0].cells).lower()
    for obsolete in ['causing input','causing heat','do not reference benv0092','after slide 17','after 17','slides 35–38','slide 34']:
        assert obsolete not in text,obsolete
    assert 'one_step_lead_rmse' not in '\n'.join(c.source for c in pair[0].cells[:ids.index('w62-task-f')])
    return pair


def execute(notebook,kernel_root,path,checks,refresh_figures=None):
    executed=copy.deepcopy(notebook)
    extra=''
    if refresh_figures is not None:
        extra='\nFIGURE_EXPORT_IDS='+repr(set(refresh_figures))+'\nEXTRA_EXPORT_NAMES={"extra_house_raw_signals"}'
        next(c for c in executed.cells if c.id=='w62-setup').source+=extra
    executed.cells.append(nbformat.v4.new_code_cell(checks+'\nPath('+repr(str(path))+').write_text(json.dumps(validation_payload,indent=2),encoding="utf8")'))
    manager=KernelManager(kernel_name='week6-day2-validation',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client=NotebookClient(executed,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}})
    try:client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    executed.cells.pop()
    if extra:
        setup=next(c for c in executed.cells if c.id=='w62-setup')
        assert setup.source.endswith(extra);setup.source=setup.source[:-len(extra)]
    return executed


def check_outputs():
    import pandas as pd
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week6_day2_figure_manifest.csv').fillna('')
    assert manifest.id.tolist()==[row[0] for row in FIGURES] and manifest.id.is_unique
    for row in manifest.itertuples():
        assert row.slide==next(spec[2] for spec in FIGURES if spec[0]==row.id)
        for path in [row.png,row.svg]:assert (ROOT/path).is_file()
        with Image.open(ROOT/row.png) as im:assert im.width>1200 and abs(im.info['dpi'][0]-220)<.1
        assert '<text' in (ROOT/row.svg).read_text(encoding='utf8')
        source=row.source_files.split(';') if row.source_files else []
        assert ';'.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source)==row.source_sha256
    tasks=pd.read_csv(ROOT/'teaching/week6_day2_task_manifest.csv')
    assert tasks.task_id.tolist()==list('ABCDEF') and tasks.minutes_including_debrief.sum()==64
    assert tasks.slide.tolist()==[row[2] for row in TASKS]
    schedule=pd.read_csv(ROOT/'teaching/week6_day2_schedule.csv')
    assert schedule.core_minutes.tolist()==SLIDE_MINUTES
    assert schedule.groupby('block').core_minutes.sum().tolist()==[50]*4
    supplemental=pd.read_csv(ROOT/'teaching/week6_day2_supplemental_placements.csv')
    assert supplemental.slide.tolist()==[27]
    for row in supplemental.itertuples():
        for name in [row.png,row.svg]:assert (ROOT/name).is_file()
        assert ';'.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in row.source_files.split(';'))==row.source_sha256
    frozen=json.loads((OUTPUT/'protected_hashes.json').read_text(encoding='utf8'))
    for name,digest in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return len(frozen)


def check_revision_baseline(directory):
    """Compare all pre-revision exported table columns, including rollout records."""
    import pandas as pd
    checked=[]
    for old in sorted(directory.glob('*.csv')):
        current=OUTPUT/old.name
        assert current.is_file(),old.name
        before=pd.read_csv(old);after=pd.read_csv(current)
        assert set(before.columns)<=set(after.columns),old.name
        pd.testing.assert_frame_equal(before,after[before.columns],check_exact=False,rtol=1e-12,atol=1e-12,obj=old.name)
        checked.append(old.name)
    assert 'multi_origin_records.csv' in checked and 'optional_fit_period_sensitivity.csv' in checked
    assert json.loads((directory/'model_provenance.json').read_text())==json.loads((OUTPUT/'model_provenance.json').read_text())
    return checked


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-dir',type=Path,help='Private pre-edit output snapshot for regression comparison.')
    parser.add_argument('--refresh-figures',nargs='*',help='Regenerate only these canonical IDs and the raw-house overview.')
    args=parser.parse_args()
    student,solution=static_checks()
    temporary=Path(tempfile.mkdtemp(prefix='BENV0169_week6_day2_'))
    kernel_root=temporary/'kernels';spec=kernel_root/'week6-day2-validation';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 6.2 validation','language':'python'}),encoding='utf8')
    executed=execute(solution,kernel_root,temporary/'solution_checks.json',CHECKS,args.refresh_figures)
    nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Fresh-kernel solution and independent numerical checks passed.',flush=True)
    completed=copy.deepcopy(student)
    for a,b in zip(completed.cells,solution.cells):
        if any(tag in a.metadata.get('tags',[]) for tag in ['task','optional_task']):a.source=b.source
    private=execute(completed,kernel_root,temporary/'completed_checks.json',CHECKS,args.refresh_figures)
    nbformat.write(private,temporary/'completed_student_PRIVATE.ipynb')
    first=json.loads((temporary/'solution_checks.json').read_text(encoding='utf8'))
    second=json.loads((temporary/'completed_checks.json').read_text(encoding='utf8'))
    assert first==second
    print('Privately completed student copy matches.',flush=True)
    wanted={'w62-setup','w62-helpers','w62-metrics','w62-house-data','w62-house-transitions','w62-house-predict','w62-house-transfer-figure'}
    demo=copy.deepcopy(student);demo.cells=[c for c in demo.cells if c.id in wanted]
    next(c for c in demo.cells if c.id=='w62-setup').source+='\nEXPORT_ARTEFACTS=False'
    execute(demo,kernel_root,temporary/'standalone_house_checks.json',DEMO_CHECKS)
    independent=json.loads((temporary/'standalone_house_checks.json').read_text(encoding='utf8'))
    for key in ['house_model','house_scores']:assert independent[key]==first[key]
    frozen_count=check_outputs();static_checks()
    baseline_tables=check_revision_baseline(args.baseline_dir) if args.baseline_dir else []
    first.update({'status':'PASS','fresh_kernel_runs':3,'protected_files_unchanged':frozen_count,
        'standalone_house_demo':'PASS: no synthetic tasks or optional exercise dependencies',
        'private_evidence_directory':str(temporary),'command':__import__('subprocess').list2cmdline([sys.executable,'-B',*sys.argv]),
        'baseline_directory':str(args.baseline_dir) if args.baseline_dir else None,'unchanged_baseline_tables':baseline_tables,
        'refreshed_figures':args.refresh_figures if args.refresh_figures is not None else 'all'})
    (OUTPUT/'validation_results.json').write_text(json.dumps(first,indent=2),encoding='utf8')
    print(f'All 11 figure pairs verified; {frozen_count} protected hashes unchanged.',flush=True)
    print(json.dumps({key:first[key] for key in ['parameters','validation_scores','house_model','house_scores','house_segments']},indent=2))


if __name__=='__main__':main()
