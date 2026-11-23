"""Scoped Week 8 control, information checks and simulation support.

The plant and data sources remain independent of the controller. Only current
nowcasts, issue-specific future forecasts and announced schedules enter plans.
"""
from dataclasses import replace
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp
from .reference_building import TwoStateReferencePlant, ReferenceBuildingParameters

ACCEPTANCE_TOL=1e-6
COMFORT_COUNT_TOL=1e-6
STRATEGIES=('Thermostat','Fixed schedule','MPC')


def solver_configuration():
    solver=next((s for s in ('CLARABEL','SCIPY','OSQP','SCS') if s in cp.installed_solvers()),None)
    if solver is None:raise RuntimeError('Install an open-source CVXPY solver in the course environment.')
    options={'tol_gap_abs':1e-9,'tol_gap_rel':1e-9,'tol_feas':1e-9} if solver=='CLARABEL' else {}
    return solver,options


def announced_schedule(operation):
    """A controller-facing table that contains no future realised weather/gains."""
    return operation[['import_price_gbp_per_kwh','comfort_min_c','comfort_max_c']].copy()


def forecast_rows(forecasts,origin,N,dt=.5):
    origin=pd.Timestamp(origin)
    rows=forecasts.loc[(forecasts.issue_time==origin)&forecasts.lead_steps.between(1,N-1)].copy()
    rows=rows.sort_values('lead_steps')
    if len(rows)!=N-1 or rows.lead_steps.tolist()!=list(range(1,N)):
        raise ValueError(f'Missing or duplicate forecast leads at {origin}; the horizon is not shortened.')
    expected=pd.DatetimeIndex(origin+pd.to_timedelta(np.arange(1,N)*dt,unit='h'))
    if not pd.DatetimeIndex(rows.valid_time).equals(expected):raise ValueError('Forecast issue/lead/valid times do not align.')
    if not np.isfinite(rows.outdoor_temperature_forecast_c).all():raise ValueError('Nonfinite future forecast.')
    return rows.reset_index(drop=True)


def schedule_horizon(announced,origin,N,dt=.5):
    times=pd.date_range(pd.Timestamp(origin),periods=N,freq=pd.Timedelta(hours=dt))
    if not announced.index.is_unique or not times.isin(announced.index).all():
        raise ValueError('Announced tariff/comfort schedules do not cover the full horizon.')
    horizon=announced.loc[times].copy()
    if not np.isfinite(horizon.to_numpy()).all():raise ValueError('Nonfinite announced schedule.')
    if np.any(horizon.comfort_min_c>horizon.comfort_max_c):raise ValueError('Inverted comfort bounds.')
    return horizon


def validate_weather_vector(outdoor,origin,nowcast,future,N,dt=.5):
    values=np.asarray(outdoor,dtype=float)
    expected=np.r_[float(nowcast),future.outdoor_temperature_forecast_c.to_numpy()]
    if values.shape!=(N,) or not np.isfinite(values).all():raise ValueError('Expected N finite outdoor inputs.')
    if not np.array_equal(values,expected):raise ValueError('Use the current nowcast followed by the same issue\'s future forecasts.')
    valid=pd.date_range(pd.Timestamp(origin),periods=N,freq=pd.Timedelta(hours=dt))
    return pd.DataFrame({'issue_time':pd.Timestamp(origin),'lead_steps':np.arange(N),'valid_time':valid,
        'outdoor_input_c':values,'source':['Current nowcast']+['Future forecast']*(N-1)})


def prepare_inputs(announced,forecasts,origin,nowcast,measurement,N=48,dt=.5,future_bias_c=0.):
    future=forecast_rows(forecasts,origin,N,dt)
    future['outdoor_temperature_forecast_c']+=float(future_bias_c)
    outdoor=np.r_[float(nowcast),future.outdoor_temperature_forecast_c.to_numpy()]
    provenance=validate_weather_vector(outdoor,origin,nowcast,future,N,dt)
    schedule=schedule_horizon(announced,origin,N,dt)
    return {'origin':pd.Timestamp(origin),'measurement_c':float(measurement),'outdoor':outdoor.copy(),
        'price':schedule.import_price_gbp_per_kwh.to_numpy(copy=True),
        'lower':schedule.comfort_min_c.to_numpy(copy=True),'upper':schedule.comfort_max_c.to_numpy(copy=True),
        'provenance':provenance,'valid_time':pd.DatetimeIndex(schedule.index)}


def make_controller(model,N=48,gamma=25.,comfort_margin_c=0.):
    """Reviewed copy of the explicit notebook LP, available for later comparisons."""
    dt=float(model['sampling_interval_hours']);b=float(model['loss_coefficient_per_step'])
    c=float(model['heat_coefficient_c_per_kw_step']);cop=float(model['nominal_cop'])
    pmax=float(model['heat_pump_max_electric_kw'])
    P=cp.Variable(N);T=cp.Variable(N+1);s=cp.Variable(N,nonneg=True)
    initial=cp.Parameter();outdoor=cp.Parameter(N);price=cp.Parameter(N,nonneg=True)
    lower=cp.Parameter(N);upper=cp.Parameter(N)
    constraints=[T[0]==initial,T[1:]==(1-b)*T[:-1]+b*outdoor+c*cop*P,
        P>=0,P<=pmax,T[1:]+s>=lower+comfort_margin_c,T[1:]<=upper-comfort_margin_c]
    problem=cp.Problem(cp.Minimize(dt*cp.sum(cp.multiply(price,P)+gamma*s)),constraints)
    return {'problem':problem,'P':P,'T':T,'s':s,'initial':initial,'outdoor':outdoor,'price':price,
        'lower':lower,'upper':upper,'N':N,'model':dict(model),'gamma':float(gamma),'comfort_margin_c':float(comfort_margin_c)}


def assign_inputs(controller,inputs):
    controller['initial'].value=inputs['measurement_c']
    for name in ('outdoor','price','lower','upper'):controller[name].value=inputs[name]


def rejected_plan(reason,status='not_solved',origin=None):
    return {'accepted':False,'status':status,'reason':str(reason),'origin':origin,
            'update_wall_s':0.,'solver_time_s':np.nan,'solve_attempted':False}


def solve_update(controller,inputs,failure=None):
    """Reject unsuccessful/stale/nonfinite candidates before exposing an action."""
    started=time.perf_counter();problem=controller['problem'];N=controller['N']
    record=rejected_plan('',origin=inputs['origin'])
    def fail(reason,status='invalid_inputs'):
        return {**record,'reason':str(reason),'status':status,'update_wall_s':time.perf_counter()-started}
    try:
        if failure is not None:return fail('Injected unusable solve: '+str(failure),'injected_failure')
        if not problem.is_dcp():return fail('Problem is not DCP.','not_dcp')
        if not np.isfinite(inputs['measurement_c']):return fail('Missing current measurement.')
        if controller['initial'].value is None or float(controller['initial'].value)!=inputs['measurement_c']:
            return fail('Initial-state parameter does not match the current observation.')
        for name in ('outdoor','price','lower','upper'):
            values=np.asarray(controller[name].value,dtype=float)
            if values.shape!=(N,) or not np.isfinite(values).all() or not np.array_equal(values,inputs[name]):
                return fail('Current '+name+' parameter is missing, stale or misaligned.')
        solver,options=solver_configuration()
        record['solve_attempted']=True
        problem.solve(solver=solver,warm_start=True,**options)
    except Exception as error:return fail(error,'solver_exception')
    record['status']=problem.status
    record['solver_time_s']=float(problem.solver_stats.solve_time or 0.) if problem.solver_stats is not None else np.nan
    if problem.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:return fail('No accepted optimal candidate.',problem.status)
    arrays={}
    for name,size in [('P',N),('T',N+1),('s',N)]:
        value=controller[name].value
        if value is None:return fail('No candidate '+name+'.',problem.status)
        array=np.asarray(value,dtype=float)
        if array.shape!=(size,) or not np.isfinite(array).all():return fail('Invalid '+name+' array.',problem.status)
        arrays[name]=array.copy()
    if problem.value is None or not np.isfinite(problem.value):return fail('Nonfinite objective.',problem.status)
    p,t,slack=arrays['P'],arrays['T'],arrays['s'];model=controller['model']
    b=model['loss_coefficient_per_step'];c=model['heat_coefficient_c_per_kw_step'];cop=model['nominal_cop']
    dt=model['sampling_interval_hours'];pmax=model['heat_pump_max_electric_kw'];margin=controller['comfort_margin_c']
    checks={'initial_c':abs(t[0]-inputs['measurement_c']),
        'dynamics_c':float(np.max(np.abs(t[1:]-(1-b)*t[:-1]-b*inputs['outdoor']-c*cop*p))),
        'power_bounds_kw':max(0.,-p.min(),p.max()-pmax),'slack_nonnegative_c':max(0.,-slack.min()),
        'lower_c':max(0.,np.max(inputs['lower']+margin-t[1:]-slack)),
        'upper_c':max(0.,np.max(t[1:]-inputs['upper']+margin))}
    bill=float(dt*np.dot(inputs['price'],p));penalty=float(dt*controller['gamma']*slack.sum())
    checks['objective_gbp']=abs(float(problem.value)-bill-penalty)
    if any(value>ACCEPTANCE_TOL for value in checks.values()):return fail('Candidate failed numerical acceptance checks.',problem.status)
    inaccurate=problem.status==cp.OPTIMAL_INACCURATE
    if inaccurate:warnings.warn('Inaccurate solver status accepted only after declared residual checks.')
    return {**record,'accepted':True,'reason':'','power_kw':p,'temperature_c':t,'slack_c':slack,
        'bill_gbp':bill,'penalty_gbp':penalty,'objective_gbp':float(problem.value),'checks':checks,
        'inaccurate_warning':inaccurate,'update_wall_s':time.perf_counter()-started,
        'outdoor':inputs['outdoor'].copy(),'lower':inputs['lower'].copy(),'upper':inputs['upper'].copy(),
        'price':inputs['price'].copy(),'valid_time':inputs['valid_time'].copy(),
        'state_time':pd.date_range(inputs['origin'],periods=N+1,freq=pd.Timedelta(hours=dt))}


def first_action(plan):
    if not plan['accepted']:raise RuntimeError('An unusable plan has no action to apply.')
    return float(plan['power_kw'][0])


def midpoint_action(measurement,lower,upper,pmax):
    if not np.isfinite(measurement):raise ValueError('The thermostat needs an available temperature.')
    return float(pmax if measurement<(lower+upper)/2 else 0.)


class FallbackPolicy:
    def __init__(self):self.consecutive_missing=0
    def decide(self,measurement,lower,upper,pmax,last_command=None,reason='unusable solve'):
        if not np.isfinite(measurement):
            self.consecutive_missing+=1
            if self.consecutive_missing>1:
                return {'abort':True,'fallback':True,'reason':'Consecutive missing temperature readings.','action_kw':None}
            command=0. if last_command is None else float(last_command)
            if not np.isfinite(command) or command<0 or command>pmax:raise ValueError('Previous command is not bounded.')
            return {'abort':False,'fallback':True,'reason':'One missing temperature: hold previous bounded command.','action_kw':command}
        self.consecutive_missing=0
        return {'abort':False,'fallback':True,'reason':str(reason),
                'action_kw':midpoint_action(measurement,lower,upper,pmax)}


def choose_command(plan,measurement,lower,upper,pmax,policy,last_command=None):
    if plan['accepted']:
        policy.consecutive_missing=0
        return {'action_kw':first_action(plan),'fallback':False,'abort':False,'reason':''}
    return policy.decide(measurement,lower,upper,pmax,last_command,plan['reason'])


class SimulationSession:
    """One physical trajectory; controllers receive observations and current inputs.

    Evaluator truth is retained separately and returned only as completed logs.
    No method obtains an extra plant observation between transitions.
    """
    def __init__(self,operation,model,day='2025-02-03',M=48,seed=88,sensor_bias_c=0.,
                 plant_resistance_multiplier=1.,missing_readings=(),plant_class=TwoStateReferencePlant):
        self._operation=operation.copy();self.model=dict(model);self.dt=float(model['sampling_interval_hours'])
        self.times=pd.date_range(pd.Timestamp(day),periods=M,freq=pd.Timedelta(hours=self.dt))
        if not operation.index.is_unique or not self.times.isin(operation.index).all():raise ValueError('Incomplete reporting day.')
        self.M=M;self.seed=seed;self.sensor_bias_c=float(sensor_bias_c);self.missing_readings=set(missing_readings)
        parameters=replace(ReferenceBuildingParameters(),resistance_outdoor_fabric_k_per_kw=
            ReferenceBuildingParameters().resistance_outdoor_fabric_k_per_kw*plant_resistance_multiplier)
        self._plant=plant_class(parameters=parameters,step_hours=self.dt,random_state=seed)
        self.capacity=float(self._plant.parameters.heat_pump_max_electric_kw)
        if self.capacity!=float(model['heat_pump_max_electric_kw']):raise ValueError('Plant/controller actuator capacity mismatch.')
        self._raw_measurement=float(self._plant.reset(20.,20.));self.position=0;self.started=time.perf_counter()
        self.intervals=[];self.boundaries=[];self.plans=[];self.forecast_records=[];self.abort_reason=''
        self._append_boundary(20.,20.)

    @property
    def measured_temperature(self):
        return np.nan if self.position in self.missing_readings else self._raw_measurement+self.sensor_bias_c

    def _append_boundary(self,true,fabric):
        timestamp=self.times[0]+pd.Timedelta(hours=self.position*self.dt)
        self.boundaries.append({'state_time':timestamp,'true_temperature_c':float(true),
            'raw_measured_temperature_c':self._raw_measurement,'measured_temperature_c':self.measured_temperature,
            'fabric_temperature_c':float(fabric)})

    def available(self):
        if self.position>=self.M:raise RuntimeError('The reporting day has ended.')
        timestamp=self.times[self.position];row=self._operation.loc[timestamp]
        return {'origin':timestamp,'measurement_c':self.measured_temperature,
            'nowcast_c':float(row.outdoor_temperature_c),'lower_c':float(row.comfort_min_c),
            'upper_c':float(row.comfort_max_c)}

    def inputs(self,announced,forecasts,N=48,future_bias_c=0.):
        view=self.available()
        inputs=prepare_inputs(announced,forecasts,view['origin'],view['nowcast_c'],view['measurement_c'],N,self.dt,future_bias_c)
        self.forecast_records.append(inputs['provenance'].assign(price=inputs['price'],lower=inputs['lower'],upper=inputs['upper']))
        return inputs

    def advance(self,action_kw,plan=None,decision=None,update_wall_s=0.,prediction_index=0):
        view=self.available();origin=view['origin'];requested=float(action_kw)
        if not np.isfinite(requested) or requested < -ACCEPTANCE_TOL or requested > self.capacity+ACCEPTANCE_TOL:
            raise ValueError('Invalid actuator command; the plant was not advanced.')
        bounded=min(self.capacity,max(0.,requested))
        correction=bounded-requested  # only numerical excess within the acceptance tolerance
        realised=self._operation.loc[origin]
        response=self._plant.step(bounded/self.capacity,float(realised.outdoor_temperature_c),
            float(realised.solar_irradiance_w_m2),float(realised.internal_gains_kw))
        self._raw_measurement=float(response['indoor_temperature_measured_c']);self.position+=1
        self._append_boundary(response['indoor_temperature_true_c'],response['fabric_temperature_c'])
        accepted=plan is not None and plan['accepted'];decision=decision or {'fallback':False,'reason':''}
        solve_attempted=bool(plan is not None and plan.get('solve_attempted',False) and plan.get('origin')==origin)
        predicted=float(plan['temperature_c'][prediction_index+1]) if accepted else np.nan
        row={'interval_start':origin,'interval_end':origin+pd.Timedelta(hours=self.dt),
            'requested_power_kw':requested,'applied_power_kw':float(response['heat_pump_electric_kw']),
            'numerical_command_correction_kw':correction,'measured_start_c':view['measurement_c'],
            'measured_end_c':self.measured_temperature,'true_end_c':float(response['indoor_temperature_true_c']),
            'predicted_next_c':predicted,'prediction_lead':prediction_index+1 if accepted else np.nan,
            'lower_c':view['lower_c'],'upper_c':view['upper_c'],'price_gbp_per_kwh':float(realised.import_price_gbp_per_kwh),
            'solver_status':plan['status'] if plan is not None else 'not_required','accepted':bool(accepted),
            'fallback':bool(decision.get('fallback',False)),'fallback_reason':decision.get('reason',''),
            'forecast_issue':plan.get('origin',pd.NaT) if plan is not None else pd.NaT,
            'forecast_first_valid':plan['valid_time'][0] if accepted else pd.NaT,
            'forecast_last_valid':plan['valid_time'][-1] if accepted else pd.NaT,
            'plan_first_power_kw':float(plan['power_kw'][prediction_index]) if accepted else np.nan,
            'solve_attempted':solve_attempted,
            'solver_time_s':plan.get('solver_time_s',np.nan) if solve_attempted else np.nan,
            'update_wall_s':float(update_wall_s),'solver_reason':plan.get('reason','') if plan is not None else '',
            'measured_minus_predicted_c':self.measured_temperature-predicted,
            'true_minus_predicted_c':float(response['indoor_temperature_true_c'])-predicted}
        if accepted:
            row.update({f'check_{key}':value for key,value in plan['checks'].items()})
            if origin.hour in (0,6,12) and origin.minute==0 and prediction_index==0:
                self.plans.append({key:(value.copy() if hasattr(value,'copy') else value) for key,value in plan.items()})
        self.intervals.append(row)
        return self.measured_temperature

    def abort(self,reason):self.abort_reason=str(reason)

    def result(self,strategy='MPC',fixed_plan_failed=False):
        return {'strategy':strategy,'intervals':pd.DataFrame(self.intervals),'boundaries':pd.DataFrame(self.boundaries),
            'plans':self.plans.copy(),'forecast_records':pd.concat(self.forecast_records,ignore_index=True) if self.forecast_records else pd.DataFrame(),
            'complete':self.position==self.M and not self.abort_reason,'abort_reason':self.abort_reason,
            'fixed_plan_failed':bool(fixed_plan_failed),'reporting_intervals':self.M,'seed':self.seed,
            'runtime_s':time.perf_counter()-self.started,'dt':self.dt,'day':str(self.times[0].date())}


def run_experiment(strategy,operation,forecasts,model,*,day='2025-02-03',M=48,N=48,seed=88,
                   gamma=25.,comfort_margin_c=0.,future_bias_c=0.,sensor_bias_c=0.,
                   plant_resistance_multiplier=1.,failures=None,missing_readings=(),
                   action_selector=choose_command,plant_class=TwoStateReferencePlant):
    """Reviewed runner used only after the notebook's visible loop task."""
    if strategy not in STRATEGIES:raise ValueError('Unknown strategy.')
    if strategy=='Fixed schedule' and N<M:raise ValueError('A committed fixed plan must cover the reporting period.')
    session=SimulationSession(operation,model,day,M,seed,sensor_bias_c,plant_resistance_multiplier,missing_readings,plant_class)
    announced=announced_schedule(operation);policy=FallbackPolicy();last=None;initial_plan=None;fixed_failed=False
    controller=make_controller(model,N,gamma,comfort_margin_c) if strategy!='Thermostat' else None
    failures=failures or {}
    for k in range(M):
        started=time.perf_counter();view=session.available();plan=None
        if strategy=='Thermostat' and np.isfinite(view['measurement_c']):
            policy.consecutive_missing=0
            decision={'action_kw':midpoint_action(view['measurement_c'],view['lower_c'],view['upper_c'],session.capacity),
                'fallback':False,'abort':False,'reason':''}
        else:
            if strategy=='Fixed schedule' and k>0 and initial_plan is not None and initial_plan['accepted']:
                plan=initial_plan
            elif strategy=='Fixed schedule' and k>0:
                plan=rejected_plan('Initial fixed plan failed; fallback operation.',origin=session.times[0])
            elif not np.isfinite(view['measurement_c']):
                plan=rejected_plan('Missing current temperature.',origin=view['origin'])
            else:
                try:
                    inputs=session.inputs(announced,forecasts,N,future_bias_c)
                    assign_inputs(controller,inputs)
                    plan=solve_update(controller,inputs,failures.get(k))
                except Exception as error:plan=rejected_plan(error,'invalid_inputs',view['origin'])
            if strategy=='Fixed schedule' and k==0:
                initial_plan=plan;fixed_failed=not plan['accepted']
            if strategy=='Thermostat':plan=rejected_plan('Missing current temperature.',origin=view['origin'])
            # A missing current observation always uses the declared missing-data policy,
            # including when a fixed plan exists; valid readings do not alter its plan.
            if not np.isfinite(view['measurement_c']):
                decision=policy.decide(view['measurement_c'],view['lower_c'],view['upper_c'],session.capacity,last,'Missing observation')
            elif strategy=='Fixed schedule' and plan['accepted']:
                policy.consecutive_missing=0
                decision={'action_kw':float(plan['power_kw'][k]),'fallback':False,'abort':False,'reason':''}
            else:decision=action_selector(plan,view['measurement_c'],view['lower_c'],view['upper_c'],session.capacity,policy,last)
        if decision['abort']:
            session.abort(decision['reason']);break
        session.advance(decision['action_kw'],plan,decision,time.perf_counter()-started,
                        prediction_index=k if strategy=='Fixed schedule' and plan is not None and plan['accepted'] else 0)
        last=session.intervals[-1]['applied_power_kw']
    return session.result(strategy,fixed_failed)


def realised_metrics(result):
    frame=result['intervals'];dt=result['dt'];complete=result['complete'] and not result['fixed_plan_failed']
    metrics={'strategy':result['strategy'],'complete':bool(complete),'completed_intervals':len(frame),
        'fallback_count':int(frame.fallback.sum()) if len(frame) else 0,
        'invalid_solve_count':int((~frame.accepted & frame.solve_attempted).sum()) if len(frame) else 0,
        'runtime_s':float(result['runtime_s']),'abort_reason':result['abort_reason'],'fixed_plan_failed':result['fixed_plan_failed']}
    keys=['energy_kwh','bill_gbp','peak_kw','below_degree_hours','above_degree_hours','total_degree_hours',
          'violation_hours','max_deviation_c','mean_endpoint_temperature_c','final_temperature_c']
    if not complete:return {**metrics,**dict.fromkeys(keys,np.nan)}
    power=frame.applied_power_kw.to_numpy();t=frame.true_end_c.to_numpy()
    below=np.maximum(frame.lower_c.to_numpy()-t,0.);above=np.maximum(t-frame.upper_c.to_numpy(),0.)
    metrics.update(energy_kwh=float(dt*power.sum()),bill_gbp=float(dt*np.dot(frame.price_gbp_per_kwh,power)),
        peak_kw=float(power.max()),below_degree_hours=float(dt*below.sum()),above_degree_hours=float(dt*above.sum()),
        total_degree_hours=float(dt*(below+above).sum()),violation_hours=float(dt*np.sum(np.maximum(below,above)>COMFORT_COUNT_TOL)),
        max_deviation_c=float(np.maximum(below,above).max()),mean_endpoint_temperature_c=float(t.mean()),final_temperature_c=float(t[-1]))
    return metrics
