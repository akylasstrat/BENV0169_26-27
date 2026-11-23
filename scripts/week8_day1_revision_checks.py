"""Private fresh-kernel checks for the 28 October Day 1 teaching changes."""

REVISION_CHECKS = r'''
from scripts.week8_day1_transfer import check_transfer_candidate as _check_transfer

# The trace has its own plant, preserves both full horizons and reuses returned readings.
assert len(guided_trace)==2 and guided_trace.accepted.all() and not guided_trace.fallback.any()
assert guided_trace.origin.tolist()==list(pd.date_range(DAY,periods=2,freq='30min'))
assert guided_trace.N.eq(48).all()
assert guided_trace.last_valid_time.tolist()==list(pd.to_datetime(['2025-02-03 23:30','2025-02-04 00:00']))
assert guided_trace.measured_initial_c.iloc[1]==guided_trace.next_measurement_c.iloc[0]
assert abs(guided_trace.measured_initial_c.iloc[1]-guided_trace.next_predicted_c.iloc[0])>1e-6
np.testing.assert_allclose(guided_trace.applied_power_kw,mpc_day['intervals'].applied_power_kw.iloc[:2],atol=1e-7,rtol=0)
np.testing.assert_allclose(guided_trace.next_measurement_c,mpc_day['intervals'].measured_end_c.iloc[:2],atol=1e-7,rtol=0)
assert trace_session is not demo
_CountedPlant.instances=[]
_trace_namespace=dict(globals())
_trace_namespace.update(display=lambda *args:None,
                        SimulationSession=_partial(SimulationSession,plant_class=_CountedPlant))
_trace_source=next(c.source for c in _public_solution.cells if c.id=='w81-guided-two-updates')
exec(compile(_trace_source,'public-two-update-loop','exec'),_trace_namespace)
_plant=_CountedPlant.instances[-1]
assert (_plant.resets,_plant.steps,_plant.observations,len(_plant.rng.draws))==(1,2,3,7)
np.testing.assert_allclose(_trace_namespace['guided_trace'].next_measurement_c,guided_trace.next_measurement_c,atol=1e-7)

# Supplied handling in the visible task preserves the reviewed failure policy.
for _missing in [{0},{1},{1,2}]:
    _visible=run_visible_mpc(operation,forecasts,model,_w8.make_controller(model,4),M=4,seed=SEED,missing_readings=_missing)
    _reviewed=_w8.run_experiment('MPC',operation,forecasts,model,M=4,N=4,seed=SEED,missing_readings=_missing)
    assert _visible['complete']==_reviewed['complete']
    assert _visible['abort_reason']==_reviewed['abort_reason']
    np.testing.assert_allclose(_visible['intervals'].applied_power_kw,_reviewed['intervals'].applied_power_kw,atol=1e-7,rtol=0)
_visible_invalid=run_visible_mpc(operation,_missing_forecasts,model,_w8.make_controller(model,4),M=3,seed=SEED)
assert _visible_invalid['complete'] and _visible_invalid['intervals'].iloc[0].fallback
assert _visible_invalid['intervals'].iloc[1].accepted

# Independent hand calculation: two intervals, original bounds, endpoint alignment.
_expected_manual=[1.,.2,2.,.5,.5]
np.testing.assert_allclose([fixture_metrics[k] for k in METRIC_KEYS],_expected_manual,atol=1e-12)
_toy_result={'strategy':'Hand fixture','complete':True,'fixed_plan_failed':False,
    'reporting_intervals':2,'dt':.5,'runtime_s':0.,'abort_reason':'',
    'intervals':metric_fixture.assign(fallback=False,accepted=True,solve_attempted=False),
    'boundaries':pd.DataFrame({'true_temperature_c':[99.,19.,23.]})}
_toy_manual=manual_metrics_for_run(_toy_result)
_toy_reviewed=realised_metrics(_toy_result)
for _key,_value in zip(METRIC_KEYS,_expected_manual):
    assert np.isclose(_toy_manual[_key],_value) and np.isclose(_toy_reviewed[_key],_value)
for _result in paired_runs.values():
    _manual=manual_metrics_for_run(_result);_reviewed=realised_metrics(_result)
    np.testing.assert_array_equal(_result['intervals'].true_end_c,_result['boundaries'].true_temperature_c.iloc[1:])
    for _key in METRIC_KEYS:assert np.isclose(_manual[_key],_reviewed[_key],rtol=0,atol=1e-10)
for _change in [{'complete':False},{'fixed_plan_failed':True},{'intervals':pd.DataFrame()},{'reporting_intervals':3}]:
    assert all(np.isnan(v) for v in manual_metrics_for_run({**_toy_result,**_change}).values())
# Student thermostat callback reproduces reviewed nominal and missing-data operation.
for _reading,_expected in [(19.,5.),(20.,0.),(21.,0.)]:
    assert thermostat_decision(_reading,18.,22.,5.)==_expected
_reviewed_thermostat=_w8.run_experiment('Thermostat',operation,forecasts,model,day=DAY,M=M,N=N,seed=SEED)
for _column in ['applied_power_kw','true_end_c','measured_end_c']:
    np.testing.assert_allclose(thermostat_day['intervals'][_column],_reviewed_thermostat['intervals'][_column],atol=1e-12,rtol=0)
for _missing in [{0},{1},{1,2}]:
    _local=run_local_thermostat(thermostat_decision,operation,model,M=4,seed=SEED,missing_readings=_missing)
    _reviewed=_w8.run_experiment('Thermostat',operation,forecasts,model,M=4,N=4,seed=SEED,missing_readings=_missing)
    assert _local['complete']==_reviewed['complete'] and _local['abort_reason']==_reviewed['abort_reason']
    np.testing.assert_array_equal(_local['intervals'].applied_power_kw,_reviewed['intervals'].applied_power_kw)
    np.testing.assert_array_equal(_local['boundaries'].measured_temperature_c,_reviewed['boundaries'].measured_temperature_c)
_callback_calls=[]
def _logged_callback(*args):
    _callback_calls.append(args)
    return thermostat_decision(*args)
_callback_result=run_local_thermostat(_logged_callback,operation,model,M=4,seed=SEED)
assert len(_callback_calls)==4
np.testing.assert_array_equal([args[0] for args in _callback_calls],_callback_result['intervals'].measured_start_c)



# Independently condensed LP for the first joint PV/battery plan (SciPy/HiGHS).
_pv_plan=pv_runs['MPC']['all_plans'][0]
_pv_forecast=_pv_plan['pv_forecast_kw']
_size=7*N+1
_slices={name:slice(i*N,(i+1)*N) for i,name in enumerate(['p','s','g','v','ch','dis'])}
_slices['e']=slice(6*N,7*N+1)
_lp_cost=np.zeros(_size)
_lp_cost[_slices['s']]=gamma*dt
_lp_cost[_slices['g']]=dt*initial_inputs['price']
_lp_cost[_slices['ch']]=_lp_cost[_slices['dis']]=throughput_weight*dt
_inequality=np.zeros((2*N,_size))
_inequality[:N,_slices['p']]=-_response
_inequality[:N,_slices['s']]=-np.eye(N)
_inequality[N:,_slices['p']]=_response
_equality=np.zeros((2*N+2,_size));_equality_rhs=np.zeros(2*N+2)
for _key,_factor in [('p',-1),('g',1),('v',-1),('ch',-1),('dis',1)]:
    _equality[:N,_slices[_key]]=_factor*np.eye(N)
_equality_rhs[:N]=-_pv_forecast
for _j in range(N):
    _equality[N+_j,6*N+_j]=-1
    _equality[N+_j,6*N+_j+1]=1
    _equality[N+_j,4*N+_j]=-dt*BATTERY['eta_charge']
    _equality[N+_j,5*N+_j]=dt/BATTERY['eta_discharge']
_equality[-2,6*N]=1;_equality[-1,-1]=1
_equality_rhs[-2:]=[BATTERY['initial_kwh'],BATTERY['target_kwh']]
_bounds=([(0,P_MAX)]*N+[(0,None)]*(2*N)+[(0,float(v)) for v in _pv_forecast]
         +[(0,BATTERY['power_limit_kw'])]*(2*N)+[(BATTERY['lower_kwh'],BATTERY['upper_kwh'])]*(N+1))
_pv_oracle=_linprog(_lp_cost,A_ub=_inequality,b_ub=_rhs,A_eq=_equality,b_eq=_equality_rhs,bounds=_bounds,method='highs')
assert _pv_oracle.success,_pv_oracle.message
np.testing.assert_allclose(_pv_plan['objective_gbp'],_pv_oracle.fun,atol=1e-6,rtol=0)

# Independent QP for the unchanged tracking example.
from scipy.optimize import minimize as _minimize, LinearConstraint as _LinearConstraint
_target=(initial_inputs['lower']+initial_inputs['upper'])/2
def _tracking_objective(x):
    error=_free+_response@x[:N]-_target
    return dt*(initial_inputs['price']@x[:N]+gamma*x[N:].sum()+lambda_tracking*(error@error))
def _tracking_gradient(x):
    error=_free+_response@x[:N]-_target
    return dt*np.r_[initial_inputs['price']+2*lambda_tracking*_response.T@error,np.full(N,gamma)]
_qp=_minimize(_tracking_objective,_oracle.x,jac=_tracking_gradient,method='SLSQP',
    bounds=[(0,P_MAX)]*N+[(0,None)]*N,
    constraints=[_LinearConstraint(_ub,-np.inf,_rhs)],options={'ftol':1e-10,'maxiter':500})
assert _qp.success,_qp.message
np.testing.assert_allclose(tracking_runs['MPC']['all_plans'][0]['objective_gbp'],_qp.fun,atol=1e-6,rtol=0)

# Every first action, input horizon, state initialisation and constraint is checked.
for _kind,_runs in [('tracking',tracking_runs),('pv_battery',pv_runs)]:
    _fixed,_mpc=_runs['Fixed plan'],_runs['MPC']
    assert set(_runs)=={'Fixed plan','MPC'}
    assert len(_fixed['all_plans'])==1 and len(_mpc['all_plans'])==48
    np.testing.assert_allclose(_fixed['intervals'].requested_power_kw,_fixed['all_plans'][0]['power_kw'],atol=1e-6)
    np.testing.assert_allclose(_fixed['all_plans'][0]['power_kw'],_mpc['all_plans'][0]['power_kw'],atol=1e-9)
    np.testing.assert_allclose(_fixed['intervals'].measured_end_c-_fixed['intervals'].true_end_c,
                              _mpc['intervals'].measured_end_c-_mpc['intervals'].true_end_c,atol=1e-12)
    for _policy,_run in _runs.items():
        assert _run['complete'] and not _run['intervals'].fallback.any()
        assert len(_run['intervals'])==48 and len(_run['boundaries'])==49
        for _i,_plan in enumerate(_run['all_plans']):
            _n=48-_i if _kind=='pv_battery' else 48
            _origin=reporting_times[_i]
            assert _plan['accepted'] and max(_plan['checks'].values())<=1e-6
            assert len(_plan['power_kw'])==_n and len(_plan['temperature_c'])==_n+1
            assert _plan['origin']==_origin
            _in=_w8.prepare_inputs(announced,forecasts,_origin,float(operation.loc[_origin,'outdoor_temperature_c']),
                                   _run['intervals'].measured_start_c.iloc[_i],N=_n,dt=dt)
            _pv=_plan.get('pv_forecast_kw')
            _kwargs={'battery':BATTERY,'energy_initial':_run['intervals'].battery_start_kwh.iloc[_i],
                     'throughput_weight':throughput_weight} if _kind=='pv_battery' else {'lambda_tracking':lambda_tracking}
            _checks,_,_=_check_transfer(_plan,_in,model,_kind,pv=_pv,gamma=gamma,**_kwargs)
            assert max(_checks.values())<=1e-6
            if _policy=='MPC':
                assert np.isclose(_run['intervals'].requested_power_kw.iloc[_i],_plan['power_kw'][0],atol=1e-6)
            if _kind=='pv_battery':
                assert _plan['valid_time'][-1]+pd.Timedelta(hours=dt)==pd.Timestamp('2025-02-04')
                _issued=pv_forecasts.loc[pv_forecasts.issue_time==_origin]
                np.testing.assert_array_equal(_pv,_issued.pv_forecast_kw)
                assert _pv[0]==pv_actual.loc[_origin]
                assert abs(_plan['energy_kwh'][-1]-5)<1e-6
        _rows=_run['intervals']
        np.testing.assert_allclose(_rows.interval_grid_cost_gbp,dt*_rows.grid_import_kw*_rows.price_gbp_per_kwh,atol=1e-12)
        if _kind=='pv_battery':
            np.testing.assert_allclose(_rows.battery_end_kwh,
                _rows.battery_start_kwh+dt*.95*_rows.charge_kw-dt*_rows.discharge_kw/.95,atol=1e-12)
            np.testing.assert_array_equal(_rows.battery_start_kwh.iloc[1:],_rows.battery_end_kwh.iloc[:-1])
            assert abs(_rows.battery_start_kwh.iloc[0]-5)<1e-12 and abs(_rows.battery_end_kwh.iloc[-1]-5)<1e-6
            assert _rows.battery_end_kwh.between(.5-1e-6,9.5+1e-6).all()
            np.testing.assert_allclose(_rows.grid_import_kw+_rows.used_pv_kw+_rows.discharge_kw,
                                       _rows.applied_power_kw+_rows.charge_kw,atol=1e-6)
            np.testing.assert_allclose(_rows.used_pv_kw+_rows.curtailed_pv_kw,_rows.realised_pv_kw,atol=1e-12)
            assert _rows.used_pv_kw.ge(0).all() and _rows.curtailed_pv_kw.ge(0).all()
            assert _rows.grid_import_kw.ge(0).all()
            assert np.minimum(_rows.charge_kw,_rows.discharge_kw).max()<1e-6
assert pv_forecast_rmse>0
assert ((future_pv.actual_pv_kw-future_pv.pv_forecast_kw).abs()>1e-3).any()
assert (pv_runs['Fixed plan']['intervals'].grid_import_kw-pv_runs['Fixed plan']['intervals'].planned_grid_import_kw).abs().max()>1e-3

# Changing later PV forecast issues cannot alter the fixed initial policy.
# Each full optional activity also runs without the other optional activities.
_public=nbformat.read(ROOT/'tutorial_8_1_model_predictive_control_solution.ipynb',4)
for _activity,_expected in [('tracking',tracking_runs),('pv',pv_runs),('shrinking',{'MPC':shrinking_run})]:
    _isolated=dict(_core_namespace)
    _isolated.update(EXPORT_ARTEFACTS=False,display=lambda *args,**kwargs:None,print=lambda *args:None)
    with _patch.object(plt,'show'):
        for _cell in _public.cells:
            if _cell.cell_type=='code' and _cell.id.startswith('w81-optional-'+_activity+'-'):
                exec(compile(_cell.source,_cell.id,'exec'),_isolated)
    _actual={'MPC':_isolated['shrinking_run']} if _activity=='shrinking' else _isolated[_activity+'_runs']
    for _name,_run in _expected.items():
        np.testing.assert_allclose(_run['intervals'].applied_power_kw,_actual[_name]['intervals'].applied_power_kw,rtol=0,atol=1e-9)
    if _activity=='pv':_pv_namespace=_isolated

# Run just the visible PV loop with later forecasts perturbed; no plotting needed.
import ast as _ast
_pv_source=next(c.source for c in _public.cells if c.id=='w81-optional-pv-run')
_pv_loop=_ast.Module(body=[_ast.parse(_pv_source).body[0]],type_ignores=[])
_later=_pv_namespace['pv_forecasts'].copy()
_mask=(_later.issue_time>pd.Timestamp(DAY)) & (_later.lead_steps>0)
_later.loc[_mask,'pv_forecast_kw']*=.6
_pv_namespace.update(pv_forecasts=_later,pv_runs={})
exec(compile(_pv_loop,'private-later-forecast-test','exec'),_pv_namespace)
np.testing.assert_allclose(_pv_namespace['pv_runs']['Fixed plan']['intervals'].applied_power_kw,
                          pv_runs['Fixed plan']['intervals'].applied_power_kw,atol=1e-9)
np.testing.assert_allclose(_pv_namespace['pv_runs']['Fixed plan']['intervals'].grid_import_kw,
                          pv_runs['Fixed plan']['intervals'].grid_import_kw,atol=1e-9)

# A rejected optimisation never advances either plant or battery.
_failed=dict(_core_namespace)
_failed.update({k:_pv_namespace[k] for k in ['BATTERY','throughput_weight','pv_forecasts','pv_actual','record_run']})
_failed['pv_runs']={}
_failed['solve_transfer']=lambda *args,**kwargs: {'accepted':False,'reason':'private injected failure','status':'infeasible'}
exec(compile(_pv_loop,'private-failed-joint-solve','exec'),_failed)
for _run in _failed['pv_runs'].values():
    assert not _run['complete'] and len(_run['intervals'])==0
    assert len(_run['boundaries'])==1
for _field,_index in [('energy_kwh',0),('energy_kwh',48),('energy_kwh',10),('charge_kw',0),('discharge_kw',0),('grid_import_kw',0)]:
    _bad=_copy.deepcopy(_pv_plan);_bad[_field][_index]+=10
    _checks,_,_=_check_transfer(_bad,initial_inputs,model,'pv_battery',pv=_pv_forecast,
        battery=BATTERY,energy_initial=5.,throughput_weight=throughput_weight)
    assert max(_checks.values())>1e-6

# The shrinking experiment changes only the planning endpoint of Task D.
assert shrinking_run['complete'] and len(shrinking_run['intervals'])==48
assert shrinking_horizons.N.tolist()==list(range(48,0,-1))
assert shrinking_horizons.endpoint.eq(pd.Timestamp('2025-02-04')).all()
np.testing.assert_allclose(shrinking_run['boundaries'].measured_temperature_c.iloc[0],
                          mpc_day['boundaries'].measured_temperature_c.iloc[0],atol=1e-12)
np.testing.assert_allclose(shrinking_run['intervals'].requested_power_kw.iloc[0],
                          mpc_day['intervals'].requested_power_kw.iloc[0],atol=1e-7)
np.testing.assert_allclose(shrinking_run['all_plans'][0]['objective_gbp'],
                          first_plan['objective_gbp'],atol=1e-6,rtol=0)
for _column in ['price_gbp_per_kwh','lower_c','upper_c']:
    np.testing.assert_array_equal(shrinking_run['intervals'][_column],mpc_day['intervals'][_column])
np.testing.assert_allclose(
    shrinking_run['intervals'].measured_end_c-shrinking_run['intervals'].true_end_c,
    mpc_day['intervals'].measured_end_c-mpc_day['intervals'].true_end_c,atol=1e-12)
for _k,_plan in enumerate(shrinking_run['all_plans']):
    _n=48-_k
    assert len(_plan['power_kw'])==_n and len(_plan['temperature_c'])==_n+1
    assert _plan['state_time'][-1]==shrinking_horizons.endpoint.iloc[_k]
    assert _plan['accepted'] and max(_plan['checks'].values())<=1e-6
    assert abs(_plan['temperature_c'][0]-shrinking_run['intervals'].measured_start_c.iloc[_k])<=1e-6
    np.testing.assert_allclose(_plan['temperature_c'][1:],
        (1-b)*_plan['temperature_c'][:-1]+b*_plan['outdoor']+c*COP*_plan['power_kw'],atol=1e-6,rtol=0)
assert shrinking_metrics.Horizon.tolist()==['Rolling MPC','Shrinking MPC']
validation_payload['shrinking_horizon']={
    'intervals':48,'horizons':shrinking_horizons.N.tolist(),
    'endpoint':'2025-02-04 00:00:00','same_initial_problem':True,
    'same_tariff_comfort_and_noise':True,'metrics':shrinking_metrics.to_dict(orient='records')}

validation_payload['revision_20261028_followup']={
    'visible_loops':'PASS: exact public Task D and two-update code; no callback planners in MPC teaching cells',
    'optional_independence':'PASS: 3 activities execute with only the core namespace',
    'pv_battery':'PASS: independent HiGHS objective, noisy causal PV forecasts, curtailment and actual grid balance, carried battery state, common 5 kWh terminal energy',
    'failures':'PASS: rejected plan advances neither building nor battery; corrupt battery candidates rejected',
    'tracking':'PASS: independent SciPy QP; core tracking behaviour retained',
    'pv_forecast_rmse_kw':pv_forecast_rmse,'joint_objective_oracle':float(_pv_oracle.fun),
    'tracking_objective_oracle':float(_qp.fun)}
'''
