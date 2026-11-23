"""Private fresh-kernel, causality, plant, controller and pairing checks for 8.1."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import sys
import uuid
import nbformat
import pandas as pd
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from build_week8_day1 import ROOT, STEM, FIGURES, TASKS, SLIDE_MINUTES, make_notebook
from week8_day1_revision_checks import REVISION_CHECKS

OUTPUT=ROOT/'teaching/week8_day1_outputs'
CHECKS=r'''
import math as _math
import copy as _copy
import warnings as _warnings
import importlib.metadata as _metadata
from unittest.mock import patch as _patch
from types import SimpleNamespace as _Namespace
from scipy.optimize import linprog as _linprog
from course_utils import week8_control as _w8
from course_utils.reference_building import TwoStateReferencePlant as _Plant

# Exercise the exact public top-level loop under private short/fault fixtures.
from functools import partial as _partial
_public_solution=nbformat.read(ROOT/'tutorial_8_1_model_predictive_control_solution.ipynb',4)
_visible_source=next(c.source for c in _public_solution.cells if c.id=='w81-task-d')
def run_visible_mpc(operation,forecasts,model,controller,M=48,seed=88,missing_readings=()):
    namespace=dict(globals())
    namespace.update(operation=operation,announced=announced_schedule(operation),
                     forecasts=forecasts,model=model,controller=controller,
                     M=M,SEED=seed,print=lambda *args: None,
                     SimulationSession=_partial(SimulationSession,missing_readings=missing_readings))
    exec(compile(_visible_source,'public-task-d','exec'),namespace)
    return namespace['mpc_day']

# Former core fault/transition exercises remain private behavioural checks.
assert Path(_w8.__file__).resolve()==ROOT/'course_utils/week8_control.py'
next_measurement=demo.advance(first_action(first_plan),first_plan)
demo_result=demo.result('Private one-transition check')
failure_run=_w8.run_experiment('MPC',operation,forecasts,model,day=DAY,M=12,N=N,seed=SEED,
    gamma=gamma,comfort_margin_c=comfort_margin_c,failures={8:'private forced failure'})
heat_loss_multiplier=.8
heat_loss_runs={name:_w8.run_experiment(name,operation,forecasts,model,day=DAY,M=M,N=N,seed=SEED,
    gamma=gamma,plant_resistance_multiplier=.8) for name in ['Fixed schedule','MPC']}
# Toy coefficients are isolated from the supplied course model.
np.testing.assert_allclose([.9*20+.1*10+.2*2,.9*19+.1*10+.2*3],[19.4,18.7],rtol=0,atol=1e-12)
assert model==json.loads((DATA_DIR/'verified_controller_model.json').read_text(encoding='utf8'))
assert b==model['loss_coefficient_per_step'] and c==model['heat_coefficient_c_per_kw_step']

assert N==M==48 and dt==.5 and gamma==25 and comfort_margin_c==0
assert SEED==88 and DAY=='2025-02-03'
assert first_view['measurement_c']!=20.
assert np.isclose(a,1-b)
assert len(forecast_alignment)==48
assert forecast_alignment.valid_time.iloc[[0,1,47]].tolist()==list(pd.to_datetime(['2025-02-03 00:00','2025-02-03 00:30','2025-02-03 23:30']))
assert first_plan['accepted'] and first_plan['status']==cp.OPTIMAL
assert np.max(np.abs(first_plan['temperature_c'][0]-first_view['measurement_c']))<=_w8.ACCEPTANCE_TOL
assert next_measurement==demo_result['boundaries'].measured_temperature_c.iloc[1]
assert len(demo_result['intervals'])==1 and len(demo_result['boundaries'])==2
np.testing.assert_allclose(demo_result['intervals'].requested_power_kw.iloc[0],first_plan['power_kw'][0],rtol=0,atol=1e-12)

# Full horizon availability at every origin, including the final reporting interval.
for _origin in reporting_times:
    _future=_w8.forecast_rows(forecasts,_origin,N,dt)
    _schedule=_w8.schedule_horizon(announced,_origin,N,dt)
    assert len(_future)==47 and len(_schedule)==48
    assert _future.issue_time.eq(_origin).all()
    assert _future.valid_time.tolist()==list(pd.date_range(_origin+pd.Timedelta(minutes=30),periods=47,freq='30min'))
assert reporting_times[-1]==pd.Timestamp('2025-02-03 23:30')
assert _schedule.index[-1]==pd.Timestamp('2025-02-04 23:00')
assert set(announced.columns)=={'import_price_gbp_per_kwh','comfort_min_c','comfort_max_c'}
_last_record=mpc_day['forecast_records'].loc[mpc_day['forecast_records'].issue_time==reporting_times[-1]]
assert len(_last_record)==48 and _last_record.valid_time.iloc[-1]==pd.Timestamp('2025-02-04 23:00')

# Two-interval hand calculation checks the model, electrical COP and time units.
_fixture_model={**model,'loss_coefficient_per_step':.1,'heat_coefficient_c_per_kw_step':.2,'nominal_cop':3.}
_fixture=_w8.make_controller(_fixture_model,2)
_fixture['problem']=cp.Problem(_fixture['problem'].objective,_fixture['problem'].constraints+[_fixture['P']==[1.,0.]])
_fixture_inputs={'origin':first_origin,'measurement_c':20.,'outdoor':np.array([10.,12.]),
    'price':np.ones(2),'lower':np.zeros(2),'upper':np.full(2,100.),
    'valid_time':pd.date_range(first_origin,periods=2,freq='30min')}
_w8.assign_inputs(_fixture,_fixture_inputs)
_fixture_plan=_w8.solve_update(_fixture,_fixture_inputs)
assert _fixture_plan['accepted']
np.testing.assert_allclose(_fixture_plan['temperature_c'],[20.,19.6,18.84],rtol=0,atol=1e-7)
assert _fixture_plan['power_kw'].shape==(2,) and _fixture_plan['temperature_c'].shape==(3,)
assert _fixture_plan['slack_c'].shape==(2,)

# A separately condensed LP verifies the first-origin lower-soft objective.
_response=np.zeros((N,N));_free=np.empty(N);_state=first_view['measurement_c']
for _k in range(N):
    _state=(1-b)*_state+b*outdoor_vector[_k];_free[_k]=_state
    for _j in range(_k+1):_response[_k,_j]=c*COP*(1-b)**(_k-_j)
_ub=np.block([[-_response,-np.eye(N)],[_response,np.zeros((N,N))]])
_rhs=np.r_[_free-initial_inputs['lower'],initial_inputs['upper']-_free]
_oracle=_linprog(np.r_[dt*initial_inputs['price'],np.full(N,dt*gamma)],A_ub=_ub,b_ub=_rhs,
    bounds=[(0,P_MAX)]*N+[(0,None)]*N,method='highs')
assert _oracle.success
np.testing.assert_allclose(first_plan['objective_gbp'],_oracle.fun,rtol=0,atol=1e-6)
np.testing.assert_allclose(first_plan['bill_gbp']+first_plan['penalty_gbp'],first_plan['objective_gbp'],rtol=0,atol=1e-6)

_checked_results={'visible MPC':mpc_day,**comparison_runs,'injected failure':failure_run,
    **{'stronger heat loss '+name:result for name,result in heat_loss_runs.items()}}
_metric_records=[]
for _name,_result in _checked_results.items():
    _frame=_result['intervals'];_states=_result['boundaries'];_length=len(_frame)
    assert _result['complete'],(_name,_result['abort_reason'])
    assert len(_states)==_length+1
    assert _states.true_temperature_c.iloc[0]==20. and _states.fabric_temperature_c.iloc[0]==20.
    assert np.isfinite(_frame.applied_power_kw).all()
    assert np.all(_frame.applied_power_kw>=0) and np.all(_frame.applied_power_kw<=P_MAX)
    assert _frame.interval_start.tolist()==list(pd.date_range(DAY,periods=_length,freq='30min'))
    assert _frame.interval_end.tolist()==_states.state_time.iloc[1:].tolist()
    np.testing.assert_array_equal(_frame.true_end_c,_states.true_temperature_c.iloc[1:])
    _accepted=_frame.loc[_frame.accepted & ~_frame.fallback]
    np.testing.assert_allclose(_accepted.requested_power_kw,_accepted.plan_first_power_kw,rtol=0,atol=1e-12)
    np.testing.assert_allclose(_frame.applied_power_kw-_frame.requested_power_kw,_frame.numerical_command_correction_kw,rtol=0,atol=1e-12)
    assert np.abs(_frame.numerical_command_correction_kw).max()<=_w8.ACCEPTANCE_TOL
    _m=realised_metrics(_result)
    _lo=[max(float(row.lower_c-row.true_end_c),0.) for row in _frame.itertuples()]
    _hi=[max(float(row.true_end_c-row.upper_c),0.) for row in _frame.itertuples()]
    _independent={'energy_kwh':dt*_math.fsum(map(float,_frame.applied_power_kw)),
        'bill_gbp':dt*_math.fsum(float(row.applied_power_kw*row.price_gbp_per_kwh) for row in _frame.itertuples()),
        'peak_kw':max(_frame.applied_power_kw),'below_degree_hours':dt*_math.fsum(_lo),
        'above_degree_hours':dt*_math.fsum(_hi),'total_degree_hours':dt*(_math.fsum(_lo)+_math.fsum(_hi)),
        'violation_hours':dt*sum(max(lo,hi)>_w8.COMFORT_COUNT_TOL for lo,hi in zip(_lo,_hi)),
        'max_deviation_c':max([*_lo,*_hi]),'mean_endpoint_temperature_c':_math.fsum(map(float,_frame.true_end_c))/_length,
        'final_temperature_c':float(_frame.true_end_c.iloc[-1])}
    for _key,_value in _independent.items():np.testing.assert_allclose(_m[_key],_value,rtol=0,atol=1e-10)
    _metric_records.append({'run':_name,**_independent,'fallback_count':_m['fallback_count'],'invalid_solve_count':_m['invalid_solve_count']})
    np.testing.assert_allclose(_frame.measured_minus_predicted_c,_frame.measured_end_c-_frame.predicted_next_c,rtol=0,atol=1e-12,equal_nan=True)
    np.testing.assert_allclose(_frame.true_minus_predicted_c,_frame.true_end_c-_frame.predicted_next_c,rtol=0,atol=1e-12,equal_nan=True)
    for _plan in _result['plans']:
        assert _plan['power_kw'].shape==(N,) and _plan['temperature_c'].shape==(N+1,)
        assert _plan['slack_c'].shape==(N,) and len(_plan['valid_time'])==N and len(_plan['state_time'])==N+1
        np.testing.assert_allclose(_plan['temperature_c'][1:],(1-b)*_plan['temperature_c'][:-1]+b*_plan['outdoor']+c*COP*_plan['power_kw'],rtol=0,atol=1e-6)
        assert np.all(_plan['temperature_c'][1:]+_plan['slack_c']>=_plan['lower']-1e-6)
        assert np.all(_plan['temperature_c'][1:]<=_plan['upper']+1e-6)
assert [p['origin'].hour for p in mpc_day['plans']]==[0,6,12]
for _p in mpc_day['plans']:
    _anchor=mpc_day['intervals'].set_index('interval_start').loc[_p['origin'],'measured_start_c']
    np.testing.assert_allclose(_p['temperature_c'][0],_anchor,rtol=0,atol=1e-6)
_frame=mpc_day['intervals']
_nowcasts=mpc_day['forecast_records'].groupby('issue_time',sort=True).first().outdoor_input_c
np.testing.assert_allclose(_frame.predicted_next_c,
    (1-b)*_frame.measured_start_c.to_numpy()+b*_nowcasts.to_numpy()+c*COP*_frame.requested_power_kw.to_numpy(),rtol=0,atol=1e-6)
assert comparison_runs['Thermostat']['intervals'].solve_attempted.sum()==0
assert comparison_runs['Fixed schedule']['intervals'].solve_attempted.sum()==1
assert comparison_runs['MPC']['intervals'].solve_attempted.sum()==48
assert comparison_runs['Fixed schedule']['intervals'].solver_time_s.iloc[1:].isna().all()
assert mpc_day['plans'][-1]['state_time'][-1]==pd.Timestamp('2025-02-04 12:00')
np.testing.assert_allclose(mpc_day['intervals'].applied_power_kw,comparison_runs['MPC']['intervals'].applied_power_kw,rtol=0,atol=1e-6)
np.testing.assert_allclose(mpc_day['boundaries'].true_temperature_c,comparison_runs['MPC']['boundaries'].true_temperature_c,rtol=0,atol=1e-6)
assert failure_run['intervals'].fallback.sum()==1
assert failure_run['intervals'].iloc[8].solver_status=='injected_failure'
assert not failure_run['intervals'].iloc[8].accepted and failure_run['intervals'].iloc[9].accepted
assert all(not result['intervals'].solver_status.str.contains('injected').any() for result in comparison_runs.values())

# The added sensitivity changes only the named reference resistance, not planning inputs.
assert heat_loss_multiplier==.8
assert model==json.loads((DATA_DIR/'verified_controller_model.json').read_text(encoding='utf8'))
np.testing.assert_array_equal(heat_loss_runs['Fixed schedule']['intervals'].applied_power_kw,
    comparison_runs['Fixed schedule']['intervals'].applied_power_kw)
for _name,_result in heat_loss_runs.items():
    assert _result['seed']==SEED and len(_result['intervals'])==M
    assert _result['boundaries'].true_temperature_c.iloc[0]==20.
    np.testing.assert_allclose(_result['plans'][0]['power_kw'],first_plan['power_kw'],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(_result['forecast_records'].outdoor_input_c,
        comparison_runs[_name]['forecast_records'].outdoor_input_c)
_params_default=_w8.ReferenceBuildingParameters()
_heat_session=_w8.SimulationSession(operation,model,M=1,plant_resistance_multiplier=.8)
from dataclasses import asdict as _asdict
_before=_asdict(_params_default);_after=_asdict(_heat_session._plant.parameters)
_changed={name for name in _before if _before[name]!=_after[name]}
assert _changed=={'resistance_outdoor_fabric_k_per_kw'}
np.testing.assert_allclose(_after['resistance_outdoor_fabric_k_per_kw'],4.8,rtol=0,atol=1e-12)

# Forecast error is an ex-post check of the actual future inputs used by MPC.
assert used_forecasts.lead_steps.ge(1).all() and (used_forecasts.valid_time<state_time[-1]).all()
assert used_forecasts.valid_time.gt(used_forecasts.issue_time).all()
assert len(used_forecasts)==sum(range(M))
_errors=operation.loc[pd.DatetimeIndex(used_forecasts.valid_time),'outdoor_temperature_c'].to_numpy()-used_forecasts.outdoor_input_c.to_numpy()
np.testing.assert_array_equal(_errors,used_forecasts.actual_minus_forecast_c)
_rmse=_math.sqrt(_math.fsum(float(x)**2 for x in _errors)/len(_errors))
np.testing.assert_allclose(forecast_error_summary.rmse_c.iloc[0],_rmse,rtol=0,atol=1e-12)
assert _rmse>0

# Private short deterministic equivalence test: reviewed runner versus visible loop.
_short_controller=_w8.make_controller(model,4,gamma,comfort_margin_c)
_visible_short=run_visible_mpc(operation,forecasts,model,_short_controller,M=4,seed=SEED)
_reviewed_short=_w8.run_experiment('MPC',operation,forecasts,model,M=4,N=4,seed=SEED)
np.testing.assert_allclose(_visible_short['intervals'].applied_power_kw,_reviewed_short['intervals'].applied_power_kw,rtol=0,atol=1e-7)
np.testing.assert_allclose(_visible_short['boundaries'].true_temperature_c,_reviewed_short['boundaries'].true_temperature_c,rtol=0,atol=1e-7)

# Stored plans are copied before the simulator receives the first command.
_advance_original=SimulationSession.advance
def _advance_mutating_plan(self,action_kw,plan=None,*args,**kwargs):
    if plan is not None and plan['accepted']:plan['power_kw'][0]+=.123
    return _advance_original(self,action_kw,plan,*args,**kwargs)
with _patch.object(SimulationSession,'advance',_advance_mutating_plan):
    _snapshot_run=run_visible_mpc(operation,forecasts,model,_w8.make_controller(model,4),M=1,seed=SEED)
np.testing.assert_allclose(_snapshot_run['plans'][0]['power_kw'],_reviewed_short['plans'][0]['power_kw'],rtol=0,atol=1e-7)

# Exact thermostat thresholds and valid-reading fallback share the Week 7 rule.
for _reading,_expected in [(19.,5.),(20.,0.),(21.,0.)]:
    assert _w8.midpoint_action(_reading,18.,22.,5.)==_expected
    assert _w8.FallbackPolicy().decide(_reading,18.,22.,5.)['action_kw']==_expected
_held=_w8.run_experiment('MPC',operation,forecasts,model,M=4,N=4,seed=SEED,missing_readings={1})
assert _held['complete'] and _held['intervals'].iloc[1].fallback and _held['intervals'].iloc[2].accepted
assert _held['intervals'].applied_power_kw.iloc[1]==_held['intervals'].applied_power_kw.iloc[0]
_aborted=_w8.run_experiment('MPC',operation,forecasts,model,M=4,N=4,seed=SEED,missing_readings={1,2})
assert not _aborted['complete'] and len(_aborted['intervals'])==2 and len(_aborted['boundaries'])==3
assert 'Consecutive missing' in _aborted['abort_reason'] and np.isnan(realised_metrics(_aborted)['bill_gbp'])
_missing_first=_w8.run_experiment('MPC',operation,forecasts,model,M=2,N=2,missing_readings={0})
assert _missing_first['intervals'].applied_power_kw.iloc[0]==0.
_fixed_failed=_w8.run_experiment('Fixed schedule',operation,forecasts,model,M=3,N=3,failures={0:'forced'})
assert _fixed_failed['fixed_plan_failed'] and not realised_metrics(_fixed_failed)['complete']
assert np.isnan(realised_metrics(_fixed_failed)['energy_kwh'])
_fixed_missing=_w8.run_experiment('Fixed schedule',operation,forecasts,model,M=2,N=2,missing_readings={0})
assert _fixed_missing['fixed_plan_failed']

# Noise innovations: instrument RNG calls without extra plant observations.
class _RecordedRNG:
    def __init__(self,rng):self.rng=rng;self.draws=[]
    def normal(self,*args,**kwargs):
        value=self.rng.normal(*args,**kwargs);self.draws.append(float(value));return value
class _CountedPlant(_Plant):
    instances=[]
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.rng=_RecordedRNG(self.rng);self.resets=0;self.steps=0;self.observations=0
        self.__class__.instances.append(self)
    def reset(self,*args,**kwargs):self.resets+=1;return super().reset(*args,**kwargs)
    def step(self,*args,**kwargs):self.steps+=1;return super().step(*args,**kwargs)
    def observe(self):self.observations+=1;return super().observe()
_noise_runs=[_w8.run_experiment(strategy,operation,forecasts,model,M=4,N=4,seed=88,plant_class=_CountedPlant) for strategy in STRATEGIES]
_plants=_CountedPlant.instances
assert len(_plants)==3
for _plant in _plants:
    assert _plant.resets==1 and _plant.steps==4 and _plant.observations==5 and len(_plant.rng.draws)==13
for _plant in _plants[1:]:np.testing.assert_array_equal(_plant.rng.draws,_plants[0].rng.draws)

# Unavailable future truth cannot influence the current command.
_baseline_current=_w8.run_experiment('MPC',operation,forecasts,model,M=1,N=48,seed=88)
_changed_truth=operation.copy();_changed_truth.loc[pd.Timestamp(DAY)+pd.Timedelta(hours=4),'outdoor_temperature_c']=-999.
_causal_current=_w8.run_experiment('MPC',_changed_truth,forecasts,model,M=1,N=48,seed=88)
np.testing.assert_array_equal(_baseline_current['intervals'].requested_power_kw,_causal_current['intervals'].requested_power_kw)
np.testing.assert_array_equal(_baseline_current['boundaries'].true_temperature_c,_causal_current['boundaries'].true_temperature_c)
_changed_nowcast=operation.copy();_changed_nowcast.loc[pd.Timestamp(DAY),'outdoor_temperature_c']-=1.
_nowcast_inputs=_w8.SimulationSession(_changed_nowcast,model,M=1).inputs(announced,forecasts,N)
assert _nowcast_inputs['outdoor'][0]==outdoor_vector[0]-1
np.testing.assert_array_equal(_nowcast_inputs['outdoor'][1:],outdoor_vector[1:])

# Fixed plan ignores subsequent observations and forecast issues after acceptance.
class _LaterMeasurementShift(_Plant):
    def step(self,*args,**kwargs):
        result=super().step(*args,**kwargs);result['indoor_temperature_measured_c']+=100.;return result
_later_forecasts=forecasts.copy()
_later_forecasts.loc[_later_forecasts.issue_time>pd.Timestamp(DAY),'outdoor_temperature_forecast_c']=np.nan
_fixed_reference=_w8.run_experiment('Fixed schedule',operation,forecasts,model,M=4,N=4,seed=88)
_fixed_changed=_w8.run_experiment('Fixed schedule',operation,_later_forecasts,model,M=4,N=4,seed=88,plant_class=_LaterMeasurementShift)
np.testing.assert_array_equal(_fixed_reference['intervals'].applied_power_kw,_fixed_changed['intervals'].applied_power_kw)
np.testing.assert_array_equal(_fixed_reference['boundaries'].true_temperature_c,_fixed_changed['boundaries'].true_temperature_c)
np.testing.assert_allclose(comparison_runs['Fixed schedule']['plans'][0]['power_kw'],comparison_runs['MPC']['plans'][0]['power_kw'],rtol=0,atol=1e-6)

# Reject malformed forecast rows rather than padding/shrinking or reading truth.
_issue_mask=forecasts.issue_time==pd.Timestamp(DAY)
_missing_forecasts=forecasts.drop(forecasts.loc[_issue_mask].index[0])
_bad_forecasts=forecasts.copy();_bad_forecasts.loc[_bad_forecasts.loc[_issue_mask].index[0],'valid_time']+=pd.Timedelta(minutes=30)
_duplicate_forecasts=pd.concat([forecasts,forecasts.loc[_issue_mask].iloc[[0]]],ignore_index=True)
for _bad in [_missing_forecasts,_bad_forecasts,_duplicate_forecasts]:
    try:_w8.forecast_rows(_bad,pd.Timestamp(DAY),48)
    except ValueError:pass
    else:raise AssertionError('Malformed forecasts accepted')
_invalid_forecast_run=_w8.run_experiment('MPC',operation,_missing_forecasts,model,M=3,N=4,seed=88)
assert _invalid_forecast_run['complete'] and _invalid_forecast_run['intervals'].iloc[0].fallback
assert _invalid_forecast_run['intervals'].iloc[1].accepted

# Status, exception, NaN, inaccurate and stale-Parameter fixtures are private.
_accepted_copy=_copy.deepcopy(first_plan)
_test=_w8.make_controller(model,4)
_test_inputs=_w8.prepare_inputs(announced,forecasts,first_origin,first_view['nowcast_c'],first_view['measurement_c'],4)
_w8.assign_inputs(_test,_test_inputs);_valid=_w8.solve_update(_test,_test_inputs);assert _valid['accepted']
with _patch.object(_test['problem'],'solve',side_effect=RuntimeError('forced exception')):
    _exception=_w8.solve_update(_test,_test_inputs)
assert not _exception['accepted'] and _exception['status']=='solver_exception' and 'power_kw' not in _exception
_test['initial'].value+=1
_stale=_w8.solve_update(_test,_test_inputs)
assert not _stale['accepted'] and 'power_kw' not in _stale
_w8.assign_inputs(_test,_test_inputs)
_bad_inputs={**_test_inputs,'lower':np.full(4,-200.),'upper':np.full(4,-100.)}
_w8.assign_inputs(_test,_bad_inputs);_infeasible=_w8.solve_update(_test,_bad_inputs)
assert not _infeasible['accepted'] and _infeasible['status']==cp.INFEASIBLE and 'power_kw' not in _infeasible
class _FakeProblem:
    status=cp.OPTIMAL_INACCURATE
    value=_valid['objective_gbp']
    solver_stats=_Namespace(solve_time=0.)
    def is_dcp(self):return True
    def solve(self,**kwargs):pass
_fake={**_test,'problem':_FakeProblem(),'P':_Namespace(value=_valid['power_kw'].copy()),
       'T':_Namespace(value=_valid['temperature_c'].copy()),'s':_Namespace(value=_valid['slack_c'].copy())}
_w8.assign_inputs(_fake,_test_inputs)
with _warnings.catch_warnings(record=True) as _caught:
    _inaccurate=_w8.solve_update(_fake,_test_inputs)
assert _inaccurate['accepted'] and _inaccurate['inaccurate_warning'] and _caught
_fake['T'].value[1]+=1.
assert not _w8.solve_update(_fake,_test_inputs)['accepted']
_fake['T'].value=_valid['temperature_c'].copy();_fake['P'].value[0]=np.nan
assert not _w8.solve_update(_fake,_test_inputs)['accepted']
for _key in ['power_kw','temperature_c','slack_c','outdoor','price','lower','upper']:
    np.testing.assert_array_equal(first_plan[_key],_accepted_copy[_key])
    assert not np.shares_memory(first_plan[_key],_valid[_key])

# Tiny command corrections are explicit; material errors do not advance the plant.
_command_session=_w8.SimulationSession(operation,model,M=1)
try:_command_session.advance(-.1)
except ValueError:pass
else:raise AssertionError('Material invalid command was applied')
assert _command_session.position==0
_command_session.advance(-.5e-6)
assert _command_session.intervals[0]['applied_power_kw']==0.
assert _command_session.intervals[0]['numerical_command_correction_kw']==.5e-6

# Public student scaffolds stop informatively without publishing completed copies.
_public_student=nbformat.read(ROOT/'tutorial_8_1_model_predictive_control.ipynb',4)
for _cell in _public_student.cells:
    if 'task' not in _cell.metadata.get('tags',[]):continue
    if _cell.cell_type=='markdown':
        assert 'TODO' in _cell.source
        continue
    try:exec(_cell.source,dict(globals()))
    except NotImplementedError as _error:assert 'Task '+_cell.metadata.tags[-1] in str(_error)
    else:raise AssertionError('Task scaffold did not stop: '+_cell.id)

# Independent daily cost and averaging checks, including the half-hour duration.
for _name,_result in paired_runs.items():
    _rows=_result['intervals'];_p=_rows.applied_power_kw.to_numpy()
    _price=_rows.price_gbp_per_kwh.to_numpy()
    _energy=.5*_math.fsum(_p);_cost=.5*_math.fsum(_p*_price)
    _record=daily_cost_summary.set_index('Policy').loc[_name]
    np.testing.assert_allclose(_record.to_numpy(dtype=float),
        [_energy,_cost,_cost/48,_cost/24,_cost/_energy],atol=1e-10)
    _evidence=daily_cost_intervals.loc[daily_cost_intervals.policy==_name]
    np.testing.assert_allclose(_evidence.cost_gbp,.5*_p*_price,atol=1e-12)
    np.testing.assert_allclose(_evidence.cumulative_cost_gbp.iloc[-1],_cost,atol=1e-12)

validation_payload={'status':'PASS','solver':SOLVER,'packages':{p:_metadata.version(p) for p in ['cvxpy','clarabel','numpy','pandas','scipy','nbclient','matplotlib']},
    'first_origin_objective_oracle_gbp':float(_oracle.fun),'hand_recurrence_c':[20.,19.6,18.84],
    'metrics':_metric_records,'visible_reviewed_short_case':'PASS: 4 intervals, 4-step horizon',
    'matched_noise_draws_per_short_run':13,'fixed_initial_failed_flag':_fixed_failed['fixed_plan_failed'],
    'missing_observation_abort_intervals':len(_aborted['intervals']),
    'nominal_fallback_counts':{name:int(r['intervals'].fallback.sum()) for name,r in comparison_runs.items()},
    'nominal_status_counts':{name:r['intervals'].solver_status.value_counts().to_dict() for name,r in comparison_runs.items()},
    'checks':'PASS: exact forecast times; hand/LP oracles; N/N+1; endpoint scoring; one physical step/action; matched innovations; causality; fixed independence; failures; stale/NaN/inaccurate fixtures; metrics; copied arrays'}
'''


def static_checks():
    student=nbformat.read(ROOT/(STEM+'.ipynb'),4);solution=nbformat.read(ROOT/(STEM+'_solution.ipynb'),4)
    assert [c.id for c in student.cells]==[c.id for c in solution.cells]
    for n,is_solution in [(student,False),(solution,True)]:
        canonical=make_notebook(is_solution)
        assert [(c.id,c.cell_type,c.source) for c in n.cells]==[(c.id,c.cell_type,c.source) for c in canonical.cells]
        for c in n.cells:
            if c.cell_type=='code':
                compile(c.source,c.id,'exec')
                assert 'course_utils.control' not in c.source and 'course_utils.mpc' not in c.source
                if not is_solution:assert c.execution_count is None and not c.outputs
    for a,b in zip(student.cells,solution.cells):
        if 'task' not in a.metadata.get('tags',[]) and not a.id.startswith('w81-response-'):assert a.source==b.source
    import ast
    ids=[c.id for c in student.cells]
    assert not any(key.startswith('w81-rebuild') for key in ids)
    assert 'w81-reference-schematic' in ids
    assert not any('optional-peak' in key or 'aggregation' in key for key in ids)
    assert ids.index('w81-optional-tracking')<ids.index('w81-optional-pv-text')
    for key in ['w81-guided-two-updates','w81-task-d','w81-optional-tracking-run','w81-optional-pv-run','w81-optional-shrinking-run']:
        code=next(c.source for c in solution.cells if c.id==key)
        syntax=ast.parse(code)
        assert any(isinstance(n,ast.For) for n in syntax.body),key
        assert not any(isinstance(n,(ast.FunctionDef,ast.Lambda)) for n in ast.walk(syntax)),key
    tasks=[c for c in student.cells if 'task' in c.metadata.get('tags',[])]
    assert [c.metadata.tags[-1] for c in tasks]==list('ABCDEF')
    assert all(('NotImplementedError' if c.cell_type=='code' else 'TODO') in c.source for c in tasks)
    assert [c.metadata.tags[-1] for c in tasks if c.cell_type=='markdown']==['A','C']
    runner_index=next(i for i,c in enumerate(student.cells) if c.id=='w81-reviewed-runner-import')
    task_d_index=next(i for i,c in enumerate(student.cells) if c.id=='w81-task-d')
    assert runner_index>task_d_index
    assert not any('run_experiment' in c.source for c in student.cells[:task_d_index])
    assert sum(SLIDE_MINUTES)==200 and [sum(SLIDE_MINUTES[i:i+9]) for i in range(0,36,9)]==[50]*4
    return student,solution


def execute(notebook,kernels,result_path,export=True):
    work=copy.deepcopy(notebook)
    if not export:
        setup=next(i for i,c in enumerate(work.cells) if c.id=='w81-setup')+1
        work.cells.insert(setup,nbformat.v4.new_code_cell('EXPORT_ARTEFACTS=False',id='private-no-exports'))
    core_end=next(i for i,c in enumerate(work.cells) if c.id=='w81-closing')
    work.cells.insert(core_end,nbformat.v4.new_code_cell('_core_namespace=dict(globals())',id='private-core-namespace'))
    probe=r'''
_save_original=save_figure
_figure_proofs=[]
def save_figure(fig,figure_id,*args,**kwargs):
    from matplotlib.patches import StepPatch
    assert tuple(fig.get_size_inches())==(12.5,3.5)
    _figure_proofs.append({'id':figure_id,'axes':len(fig.axes),
        'steps':[len(p.get_data().edges) for ax in fig.axes for p in ax.patches if isinstance(p,StepPatch)],
        'line_lengths':[len(line.get_xdata()) for ax in fig.axes for line in ax.lines]})
    _save_original(fig,figure_id,*args,**kwargs)
'''
    position=next(i for i,c in enumerate(work.cells) if c.id=='w81-plot-helpers')+1
    work.cells.insert(position,nbformat.v4.new_code_cell(probe,id='private-figure-probe'))
    checks='import nbformat\n'+CHECKS+REVISION_CHECKS+'''\nassert len(_figure_proofs)==6
_proof={p['id']:p for p in _figure_proofs}
assert _proof['F81-01']['steps']==[49]
assert _proof['F81-02']['steps']==[49,49]
assert _proof['F81-03']['axes']==2 and _proof['F81-03']['steps']==[49,49]
assert _proof['F81-03']['line_lengths'].count(49)>=3
assert _proof['F81-03-table']['axes']==1
assert _proof['F81-04']['axes']==1
validation_payload['figure_geometry']=_figure_proofs
validation_payload['toy_next_temperatures_c']=[19.4,18.7]
validation_payload['forecast_origin_lead_pair_rmse_c']=_rmse
'''+f'\nPath({str(result_path)!r}).write_text(json.dumps(validation_payload,indent=2),encoding="utf8")'
    work.cells.append(nbformat.v4.new_code_cell(checks,id='private-validation'))
    km=KernelManager(kernel_name='week8-day1-validation',
        connection_file=str(result_path.with_suffix('.connection.json')),
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    try:
        NotebookClient(work,km=km,timeout=300,resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        # A caller-supplied KernelManager is not owned/closed by NotebookClient.
        if km.has_kernel:km.shutdown_kernel(now=True)
    work.cells=[c for c in work.cells if not c.id.startswith('private-')]
    return work


def output_checks(snapshot):
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week8_day1_figure_manifest.csv')
    assert len(manifest)==6 and set(manifest.id)=={row[0] for row in FIGURES}
    for id,slide,stem,caption,task in FIGURES:
        record=manifest.set_index('id').loc[id]
        assert record.slide==slide and record.task==task
        with Image.open(ROOT/record.png) as im:assert im.size==(2750,770)
        assert '<text' in (ROOT/record.svg).read_text(encoding='utf8')
    tasks=pd.read_csv(ROOT/'teaching/week8_day1_task_manifest.csv')
    assert tasks[['task_id','slide','minutes_including_debrief']].values.tolist()==[list(row) for row in TASKS]
    assert tasks.student_work_minutes.tolist()==[5,12,5,18,6,5]
    assert tasks.debrief_minutes.tolist()==[3,3,3,4,4,3]
    schedule=pd.read_csv(ROOT/'teaching/week8_day1_schedule.csv')
    assert schedule.slide.tolist()==list(range(1,41)) and schedule.core_minutes.tolist()==SLIDE_MINUTES+[0]*4
    assert schedule.loc[schedule.block>0].groupby('block').core_minutes.sum().tolist()==[50]*4
    assert schedule.guided_minutes.sum()==43
    assert tasks.minutes_including_debrief.sum()==71
    assert int(schedule.guided_minutes.sum()+tasks.minutes_including_debrief.sum())==114
    assert schedule.loc[schedule.guided_minutes>0,['slide','guided_minutes']].values.tolist()==[[5,4],[9,6],[17,5],[23,7],[25,4],[34,5],[35,12]]
    for prefix in ['successive_plans','fixed_vs_mpc_fixed_schedule','fixed_vs_mpc_mpc','three_policy_thermostat']:
        assert len(pd.read_csv(OUTPUT/(prefix+'_intervals.csv')))==48
        assert len(pd.read_csv(OUTPUT/(prefix+'_boundaries.csv')))==49
    assert pd.read_csv(OUTPUT/'three_policy_metrics_native_table.csv').shape==(7,4)
    concepts=pd.read_csv(OUTPUT/'concept_manifest.csv')
    assert set(concepts.id)=={f'C81-{i:02d}' for i in range(1,7)}
    for row in concepts.itertuples():
        assert '<text' in (ROOT/row.svg).read_text(encoding='utf8')
    # Numerical exports match independently scored interval evidence.
    metrics=pd.read_csv(OUTPUT/'three_policy_metrics.csv').set_index('strategy')
    for name,prefix in [('Fixed schedule','fixed_vs_mpc_fixed_schedule'),('MPC','fixed_vs_mpc_mpc'),('Thermostat','three_policy_thermostat')]:
        rows=pd.read_csv(OUTPUT/(prefix+'_intervals.csv'))
        expected={'energy_kwh':.5*math.fsum(rows.applied_power_kw),
            'bill_gbp':.5*math.fsum(rows.applied_power_kw*rows.price_gbp_per_kwh),
            'below_degree_hours':.5*math.fsum((rows.lower_c-rows.true_end_c).clip(lower=0)),
            'above_degree_hours':.5*math.fsum((rows.true_end_c-rows.upper_c).clip(lower=0))}
        for key,value in expected.items():assert math.isclose(metrics.loc[name,key],value,abs_tol=1e-10)
        # Compare with the current pre-edit evidence, excluding machine-dependent timing.
        for suffix in ['intervals','boundaries']:
            filename=prefix+'_'+suffix+'.csv'
            before=pd.read_csv(snapshot/'before/teaching/week8_day1_outputs'/filename)
            after=pd.read_csv(OUTPUT/filename)
            stable=[c for c in before.columns if c not in {'solver_time_s','update_wall_s'}]
            pd.testing.assert_frame_equal(before[stable],after[stable],check_exact=False,rtol=0,atol=1e-6)
    assets=pd.read_csv(OUTPUT/'guided_transfer_asset_manifest.csv')
    assert len(assets)==5 and set(assets.slide)=={23,34}
    current_cells={c.id for c in nbformat.read(ROOT/(STEM+'.ipynb'),4).cells}
    for row in assets.itertuples():
        assert row.producer_cell_id in current_cells
        assert hashlib.sha256((OUTPUT/row.file).read_bytes()).hexdigest()==row.sha256
        for name,digest in json.loads(row.source_sha256).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    pd.testing.assert_frame_equal(assets,pd.read_csv(ROOT/'teaching/week8_day1_supplemental_placements.csv'))
    protected=json.loads((snapshot/'protected_hashes.json').read_text(encoding='utf8'))
    for name,digest in protected.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    schematic=ROOT/'figures/week8_day1/reference_model_schematic'
    with Image.open(schematic.with_suffix('.png')) as im:assert im.size==(3200,1800)
    assert '<text' in schematic.with_suffix('.svg').read_text(encoding='utf8')
    assert schematic.with_suffix('.pdf').read_bytes().startswith(b'%PDF')
    return len(protected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args()
    student,solution=static_checks()
    private=ROOT/'tmp'/('week8_day1_validation_'+uuid.uuid4().hex)
    spec=private/'kernels/week8-day1-validation';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 8.1 validation','language':'python'}),encoding='utf8')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    executed=execute(solution,spec.parent,private/'solution_checks.json')
    nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Fresh-kernel solution passed independent checks.',flush=True)
    completed=copy.deepcopy(student)
    for a,b in zip(completed.cells,solution.cells):
        if 'task' in a.metadata.get('tags',[]):a.source=b.source
    completed=execute(completed,spec.parent,private/'student_checks.json',export=False)
    nbformat.write(completed,private/'completed_student_PRIVATE.ipynb')
    first=json.loads((private/'solution_checks.json').read_text(encoding='utf8'))
    second=json.loads((private/'student_checks.json').read_text(encoding='utf8'))
    assert first==second
    static_checks();protected=output_checks(args.snapshot)
    first.update(fresh_kernel_runs=2,paired_cells=len(student.cells),student_outputs=0,student_tasks=6,
        figure_pairs=len(FIGURES),protected_files_unchanged=protected,private_evidence_directory=str(private),
        pre_edit_snapshot=str(args.snapshot),teaching_minutes=200,task_minutes=71,guided_minutes=43,active_minutes=114,concept_illustrations=6,
        acceptance_tolerance=1e-6,comfort_count_tolerance_c=1e-6,
        nominal_pre_edit_comparison='PASS: all fixed/MPC/thermostat interval and boundary columns within 1e-6; timing excluded',
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['course_utils/week8_control.py','scripts/build_week8_day1.py','scripts/week8_day1_visuals.py','scripts/week8_review_diagnostics.py','scripts/week8_day1_transfer.py','scripts/week8_day1_revision_checks.py','scripts/week8_day1_optional.py','scripts/validate_week8_day1.py',STEM+'.ipynb',STEM+'_solution.ipynb']})
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    (OUTPUT/'validation_results.json').write_text(json.dumps(finite(first),indent=2,allow_nan=False),encoding='utf8')
    print(f'PASS: 2 fresh kernels; {len(student.cells)} paired cells; 6 numerical figures/tables; {protected} protected files unchanged.',flush=True)
    print(json.dumps(first['nominal_status_counts'],indent=2))


if __name__=='__main__':main()
