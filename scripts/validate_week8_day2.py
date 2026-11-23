"""Fresh-kernel validation of independent error injections and paired outcomes."""
from pathlib import Path
import argparse,copy,hashlib,json,math,sys,uuid
import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from build_week8_day2 import ROOT,STEM,make_notebook,FIGURES,SLIDE_MINUTES

CHECKS=r'''
import math as _math
from course_utils import week8_errors as _errors
from scripts.week8_review_diagnostics import residual_evidence
from course_utils.week8_control import COMFORT_COUNT_TOL
assert Path(_errors.__file__).resolve()==ROOT/'course_utils/week8_errors.py'
assert N==M==48 and dt==.5 and gamma==25 and comfort_margin_c==0
assert len(worked_metrics)==34 and len(final_metrics)==48 and len(final_pairs)==24
assert len(worked_runs)==34 and len(final_runs)==48
_all={**{('worked',)+key:value for key,value in worked_runs.items()},
      **{('final',)+key:value for key,value in final_runs.items()}}
_payload=[]
for _key,_result in _all.items():
    _tag,_day,_name,_seed,_strategy=_key;_case=CASES[_name];_draws=innovations(_seed)
    _frame=_result['intervals'];_states=_result['boundaries'];_records=_result['forecast_records']
    assert _result['complete'],(_key,_result['abort_reason'])
    assert len(_frame)==48 and len(_states)==49
    assert _states.true_temperature_c.iloc[0]==20.
    assert 'fabric_temperature_c' not in _states
    assert _frame.interval_end.tolist()==_states.state_time.iloc[1:].tolist()
    assert np.isfinite(_frame.applied_power_kw).all()
    _true=_states.true_temperature_c.to_numpy();_power=_frame.applied_power_kw.to_numpy()
    _b=np.array([model['loss_coefficient_per_step']*(_case.loss_multiplier if k>=_case.drift_after_step else 1.) for k in range(M)])
    _actual=operation.loc[pd.DatetimeIndex(_frame.interval_start),'outdoor_temperature_c'].to_numpy()
    _expected=(1-_b)*_true[:-1]+_b*_actual+model['heat_coefficient_c_per_kw_step']*model['nominal_cop']*_power+_case.process_std_c*_draws['process']
    np.testing.assert_allclose(_true[1:],_expected,rtol=0,atol=1e-11)
    np.testing.assert_allclose(_states.measured_temperature_c,_true+_case.sensor_bias_c+_case.sensor_std_c*_draws['sensor'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(_frame.measured_start_c,_states.measured_temperature_c.iloc[:-1],rtol=0,atol=1e-12)
    _origin_idx=((_records.issue_time-pd.Timestamp(_day))/pd.Timedelta(minutes=30)).astype(int).to_numpy()
    _lead=_records.lead_steps.to_numpy();_truth=operation.loc[pd.DatetimeIndex(_records.valid_time),'outdoor_temperature_c'].to_numpy()
    _perturb=np.where(_lead==0,0.,_case.forecast_bias_c+_case.forecast_std_c*_draws['forecast'][_origin_idx,np.maximum(_lead-1,0)])
    np.testing.assert_allclose(_records.outdoor_input_c,_truth+_perturb,rtol=0,atol=1e-12)
    assert (_records.valid_time==_records.issue_time+pd.to_timedelta(_lead*.5,unit='h')).all()
    _lo=np.maximum(_frame.lower_c.to_numpy()-_true[1:],0.);_hi=np.maximum(_true[1:]-_frame.upper_c.to_numpy(),0.)
    _expected_metrics={'energy_kwh':.5*_math.fsum(_power),
        'bill_gbp':.5*_math.fsum(_power*_frame.price_gbp_per_kwh),
        'peak_kw':float(max(_power)),'below_degree_hours':.5*_math.fsum(_lo),
        'above_degree_hours':.5*_math.fsum(_hi),'total_degree_hours':.5*_math.fsum(_lo+_hi),
        'mean_endpoint_temperature_c':_math.fsum(_true[1:])/48,'final_temperature_c':float(_true[-1]),
        'violation_hours':.5*sum(np.maximum(_lo,_hi)>COMFORT_COUNT_TOL),'max_deviation_c':float(max(np.max(_lo),np.max(_hi)))}
    _metrics=realised_metrics(_result)
    for _field,_value in _expected_metrics.items():np.testing.assert_allclose(_metrics[_field],_value,rtol=0,atol=1e-10)
    if _strategy=='MPC':
        np.testing.assert_allclose(_frame.requested_power_kw,_frame.plan_first_power_kw,rtol=0,atol=1e-12)
        assert _frame.solve_attempted.sum()==48
        assert _records.valid_time.iloc[-1]==pd.Timestamp(_day)+pd.Timedelta(hours=47)
    else:
        assert _frame.solve_attempted.sum()==1
        np.testing.assert_allclose(_frame.requested_power_kw,_result['plans'][0]['power_kw'],rtol=0,atol=1e-12)
    _payload.append({'key':list(_key),**_expected_metrics,'fallback':int(_frame.fallback.sum())})

# Same time-indexed innovations and initial information across strategies.
for _runs in [worked_runs,final_runs]:
    for (_day,_name,_seed,_strategy),_result in _runs.items():
        _other=_runs[(_day,_name,_seed,'MPC' if _strategy=='Fixed schedule' else 'Fixed schedule')]
        assert _result['innovation_sha256']==_other['innovation_sha256']
        np.testing.assert_allclose(_result['plans'][0]['power_kw'],_other['plans'][0]['power_kw'],rtol=0,atol=1e-6)

# Drift is identical to baseline before the named noon transition.
for _strategy in ['Fixed schedule','MPC']:
    _base=result_at('C0 No-error reference',_strategy);_drift=result_at('C8 Midday model drift',_strategy)
    np.testing.assert_allclose(_base['boundaries'].true_temperature_c.iloc[:25],_drift['boundaries'].true_temperature_c.iloc[:25],rtol=0,atol=1e-9)
    np.testing.assert_allclose(_base['intervals'].applied_power_kw.iloc[:24],_drift['intervals'].applied_power_kw.iloc[:24],rtol=0,atol=1e-9)

# Current j=0 inputs and actual conditions stay identical under forecast injections.
_c0=result_at('C0 No-error reference')['forecast_records']
for _case in ['C1 Forecast noise','C2 Forecast bias']:
    _rec=result_at(_case)['forecast_records']
    np.testing.assert_array_equal(_rec.actual_outdoor_c,_c0.actual_outdoor_c)
    np.testing.assert_array_equal(_rec.loc[_rec.lead_steps==0,'outdoor_input_c'],_c0.loc[_c0.lead_steps==0,'outdoor_input_c'])

# Independent samples only for stochastic cases; deterministic stream seed is irrelevant.
_det_frame,_det_runs=run_case(operation,model,'C0 No-error reference',CASES['C0 No-error reference'],WORKED_DAY,22)
for _strategy,_res in _det_runs.items():
    np.testing.assert_allclose(_res['boundaries'].true_temperature_c,result_at('C0 No-error reference',_strategy)['boundaries'].true_temperature_c,rtol=0,atol=1e-12)

# Weather coverage, case preconditions and frozen settings fail explicitly.
for _bad in [ErrorCase(loss_multiplier=100),ErrorCase(sensor_std_c=-1)]:
    try:_errors.validate_case(_bad,model)
    except ValueError:pass
    else:raise AssertionError('Invalid case accepted')
try:forecast_table(operation.iloc[:48],WORKED_DAY,CASES['C0 No-error reference'],innovations(11))
except ValueError:pass
else:raise AssertionError('Missing future coverage silently accepted')
import copy as _copy
_bad_protocol=_copy.deepcopy(frozen_protocol);_bad_protocol['gamma']=1.
try:verify_protocol(ROOT,_bad_protocol)
except ValueError:pass
else:raise AssertionError('Frozen policy changed silently')

# Incomplete runs remain missing in scoring and in outer-joined pairs.
_partial=_copy.deepcopy(result_at('C0 No-error reference'));_partial['complete']=False
assert np.isnan(realised_metrics(_partial)['bill_gbp'])
_missing=worked_metrics.loc[~((worked_metrics['case']=='C0 No-error reference')&(worked_metrics.strategy=='MPC'))]
assert not pairs(_missing).query("`case` == 'C0 No-error reference'").valid_pair.any()

# Transfer remains a separate evaluation on the richer physical model.
for _result in transfer_runs.values():
    assert _result['complete'] and _result['seed']==90
    assert 'fabric_temperature_c' in _result['boundaries']
    assert len(_result['boundaries'])==49
assert len(final_pairs)==24 and evidence['valid_pairs']==24
# Stored residuals use accepted first actions and the current exact nowcast.
for _name in ['C0 No-error reference','C1 Forecast noise','C2 Forecast bias','C5 Sensor bias']:
    _res=result_at(_name);_rows=_res['intervals'];_frame=residual_evidence(_res)
    assert len(_frame)==48
    _now=_res['forecast_records'].query('lead_steps == 0')
    np.testing.assert_array_equal(_now.outdoor_input_c,_now.actual_outdoor_c)
    np.testing.assert_allclose(_frame.measured_minus_predicted_c,
        _rows.measured_end_c-_rows.predicted_next_c,rtol=0,atol=1e-12)
    _bias=CASES[_name].sensor_bias_c
    np.testing.assert_allclose(_frame.measured_minus_predicted_c,
        model['loss_coefficient_per_step']*_bias,rtol=0,atol=1e-8)
    np.testing.assert_allclose(_frame.measured_minus_true_c,_bias,rtol=0,atol=1e-12)
# Filtering must exclude rejected, fallback, old-plan and missing predictions.
_filter=_copy.deepcopy(result_at('C0 No-error reference'));_rf=_filter['intervals']
_rf.loc[0,'accepted']=False;_rf.loc[1,'fallback']=True
_rf.loc[2,'prediction_lead']=2;_rf.loc[3,'solve_attempted']=False
_rf.loc[4,'predicted_next_c']=np.nan
assert len(residual_evidence(_filter))==43
assert residual_evidence(_filter).interval_start.iloc[0]==_rf.interval_start.iloc[5]
assert len(sensor_detail)==14 and sensor_detail.groupby('case').size().tolist()==[6,2,6]
assert final_case_evidence.valid_pairs.tolist()==[9,3,3,9]
for _name,_group in final_pairs.groupby('case',sort=False):
    _row=final_case_evidence.set_index('case').loc[_name];_valid=_group.loc[_group.valid_pair]
    assert _row.valid_pairs==len(_valid) and _row.invalid_pairs==len(_group)-len(_valid)
    assert _row.mpc_lower_bill==sum(_valid.delta_bill_gbp<0)
    np.testing.assert_allclose(_row.mean_bill_difference_gbp,_math.fsum(_valid.delta_bill_gbp)/len(_valid))
    np.testing.assert_allclose(_row.mean_discomfort_difference_degree_hours,_math.fsum(_valid.delta_total_degree_hours)/len(_valid))
assert case_evidence(pairs(_missing)).invalid_pairs.sum()==1

# C0 is a no-error check; the forecast teaching comparison is C1 versus C2.
np.testing.assert_allclose(result_at('C0 No-error reference','MPC')['intervals'].applied_power_kw,
    result_at('C0 No-error reference','Fixed schedule')['intervals'].applied_power_kw,rtol=0,atol=1e-6)
assert set(forecast_metrics['case'])=={'C1 Forecast noise','C2 Forecast bias'}
assert len(forecast_metrics)==8 and len(forecast_pairs)==4
_noise=forecast_pairs.loc[forecast_pairs['case']=='C1 Forecast noise']
assert (_noise.delta_total_degree_hours<0).all()
assert (_noise.delta_bill_gbp<0).any() and (_noise.delta_bill_gbp>0).any()
_bias=forecast_pairs.loc[forecast_pairs['case']=='C2 Forecast bias']
assert (_bias.delta_bill_gbp>0).all() and (_bias.delta_total_degree_hours<0).all()
assert {f['case'].iloc[0] for f in forecast_consequences}=={'C1 Forecast noise','C2 Forecast bias'}

# The displayed bars and whiskers are computed from the declared valid runs.
for _summary in [worked_summary,forecast_summary]:
    for _row in _summary.itertuples():
        _part=worked_metrics.loc[(worked_metrics['case']==_row.case)&(worked_metrics.strategy==_row.strategy)&worked_metrics.complete]
        assert _row.valid==len(_part)
        for _metric in ['bill_gbp','total_degree_hours']:
            for _suffix,_value in [('mean',_part[_metric].mean()),('min',_part[_metric].min()),('max',_part[_metric].max())]:
                np.testing.assert_allclose(getattr(_row,_metric+'_'+_suffix),_value,rtol=0,atol=1e-12)
for _row in final_bar_summary.itertuples():
    _part=valid_final.loc[valid_final['case']==_row.case]
    assert _row.valid_pairs==len(_part)
    for _metric in ['delta_bill_gbp','delta_total_degree_hours']:
        for _suffix,_value in [('mean',_part[_metric].mean()),('min',_part[_metric].min()),('max',_part[_metric].max())]:
            np.testing.assert_allclose(getattr(_row,_metric+'_'+_suffix),_value,rtol=0,atol=1e-12)
validation_payload={'status':'PASS','metrics':_payload,'evidence':evidence,'worked_runs':34,'final_runs':48,
    'new_diagnostics':'PASS: C1/C2 fixed-MPC contrasts; bars and observed-range statistics; exact nowcasts; sensor-bias relationship; individual seed rows; invalid pairs retained.',
    'checks':'Independent recurrence/metrics; time-indexed paired innovations; separate error injection; forecast coverage/nowcasts; initial common information; model drift; failure retention; frozen protocol; separate richer transfer.'}
from scripts.week8_stochastic_checks import check_extension
validation_payload['stochastic_extension']=check_extension(model,weather_covariance,
    sampled_errors,realised_errors,make_scenario_controller,stochastic_inputs,
    stochastic_initial_plans,tiny_plans,stochastic_metrics,stochastic_intervals,
    stochastic_saved_plans,stochastic_pairs,stochastic_summary,stochastic_evidence,stochastic_protocol)
'''

def static_checks():
    student=nbformat.read(ROOT/(STEM+'.ipynb'),4);solution=nbformat.read(ROOT/(STEM+'_solution.ipynb'),4)
    assert [c.id for c in student.cells]==[c.id for c in solution.cells]
    for n,sol in [(student,False),(solution,True)]:
        assert [(c.id,c.source) for c in n.cells]==[(c.id,c.source) for c in make_notebook(sol).cells]
        for c in n.cells:
            if c.cell_type=='code':
                compile(c.source,c.id,'exec')
                if not sol:assert not c.outputs and c.execution_count is None
    tasks=[c for c in student.cells if 'task' in c.metadata.get('tags',[])]
    assert [c.metadata.tags[-1] for c in tasks]==list('ABCDEFGH')
    assert all(('TODO' if c.cell_type=='markdown' else 'NotImplementedError') in c.source for c in tasks)
    for a,b in zip(student.cells,solution.cells):
        if 'task' not in a.metadata.get('tags',[]) and not a.id.startswith('w82-response-'):assert a.source==b.source
    assert sum(SLIDE_MINUTES)==192 and [sum(SLIDE_MINUTES[i:i+8]) for i in range(0,32,8)]==[50,50,50,42]
    ids=[c.id for c in student.cells]
    removed={'w82-project-checkpoint','w82-response-project','w82-optional','w82-margin',
             'w82-residual-text','w82-residual-figure','w82-response-residual'}
    assert removed.isdisjoint(ids) and 'w82-learning-check' in ids
    assert all('## Optional:' not in c.source and 'Group proposal checkpoint' not in c.source for c in student.cells)
    for key in ['w82-forecast-text','w82-model-text','w82-sensor-text','w82-combined-text']:
        assert r'\color{red}' in next(c.source for c in student.cells if c.id==key)
    return student,solution

def execute(notebook,kernels,destination,export=True):
    n=copy.deepcopy(notebook)
    if not export:
        i=next(i for i,c in enumerate(n.cells) if c.id=='w82-setup')+1
        n.cells.insert(i,nbformat.v4.new_code_cell('EXPORT_ARTEFACTS=False',id='private-no-export'))
    n.cells.append(nbformat.v4.new_code_cell(CHECKS+f'\nPath({str(destination)!r}).write_text(json.dumps(validation_payload,indent=2),encoding="utf8")',id='private-checks'))
    km=KernelManager(kernel_name='week8-errors',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    NotebookClient(n,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute()
    n.cells=[c for c in n.cells if not c.id.startswith('private-')]
    return n

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args()
    student,solution=static_checks();private=ROOT/'tmp'/('week8_errors_validation_'+uuid.uuid4().hex)
    spec=private/'kernels/week8-errors';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Week 8 errors','language':'python'}),encoding='utf8')
    executed=execute(solution,spec.parent,private/'solution.json');nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Day 2 solution passed fresh-kernel checks.',flush=True)
    completed=copy.deepcopy(student)
    for a,b in zip(completed.cells,solution.cells):
        if 'task' in a.metadata.get('tags',[]):a.source=b.source
    executed_student=execute(completed,spec.parent,private/'student.json',False)
    nbformat.write(executed_student,private/'completed_student_PRIVATE.ipynb')
    first=json.loads((private/'solution.json').read_text());second=json.loads((private/'student.json').read_text());assert first==second
    static_checks()
    protected=json.loads((args.snapshot/'protected_hashes.json').read_text())
    for name,digest in protected.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week8_day2_figure_manifest.csv');assert len(manifest)==len(FIGURES)==15
    assert manifest.set_index('id').loc['F82-08','seeds']=='90'
    assert manifest.cell_tag.isin([c.id for c in solution.cells]).all()
    assert pd.read_csv(ROOT/'teaching/week8_day2_schedule.csv').guided_minutes.sum()==32
    panels=pd.read_csv(ROOT/'teaching/week8_day2_outputs/sensor_panel_manifest.csv')
    assert len(panels)==3
    for row in panels.itertuples():
        with Image.open(ROOT/row.png) as im:assert im.size==(2750,770)
        assert '<text' in (ROOT/row.svg).read_text(encoding='utf8')
    for row in manifest.itertuples():
        with Image.open(ROOT/row.png) as im:assert im.width==2750 and im.height>=770
        assert '<text' in (ROOT/row.svg).read_text(encoding='utf8')
    first.update(fresh_kernel_runs=2,paired_cells=len(student.cells),student_outputs=0,figure_pairs=len(FIGURES),sensor_panel_pairs=3,
        protected_files_unchanged=len(protected),private_directory=str(private),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/build_week8_day2.py','scripts/validate_week8_day2.py','course_utils/week8_errors.py',STEM+'.ipynb',STEM+'_solution.ipynb']})
    (ROOT/'teaching/week8_day2_outputs/validation_results.json').write_text(json.dumps(first,indent=2,allow_nan=False),encoding='utf8')
    print(f'PASS: 2 fresh kernels; 34 worked + 48 final runs; 24 pairs; {len(protected)} protected files.',flush=True)
if __name__=='__main__':main()
