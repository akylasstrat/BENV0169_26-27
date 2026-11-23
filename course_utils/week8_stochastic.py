"""Supplied data preparation and execution for the introductory scenario-MPC task.

The notebook defines the optimisation model. This file supplies chronological
error alignment, numerical acceptance checks and the familiar first-action loop.
"""
import time
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp

from .week8_control import (
    ACCEPTANCE_TOL, SimulationSession, FallbackPolicy, announced_schedule,
    choose_command, rejected_plan, solver_configuration)
from .week8_errors import ErrorCase, innovations, controlled_plant


def forecast_error_archive(operation, forecasts, cutoff, N=48):
    """Return complete actual-minus-forecast vectors known before the cutoff."""
    rows = forecasts.loc[forecasts.lead_steps.between(1, N-1)].copy()
    if rows.duplicated(['issue_time', 'lead_steps']).any():
        raise ValueError('Repeated issue-time/lead pair.')
    expected = rows.issue_time + pd.to_timedelta(rows.lead_steps*.5, unit='h')
    if not rows.valid_time.equals(expected):
        raise ValueError('Forecast valid times and leads differ.')
    rows = rows.loc[(rows.valid_time < pd.Timestamp(cutoff))
                    & rows.valid_time.isin(operation.index)].copy()
    rows['error'] = (operation.loc[pd.DatetimeIndex(rows.valid_time),
                                  'outdoor_temperature_c'].to_numpy()
                     - rows.outdoor_temperature_forecast_c.to_numpy())
    archive = rows.pivot(index='issue_time', columns='lead_steps', values='error')
    archive = archive.reindex(columns=range(1, N)).dropna().sort_index()
    if archive.empty or not np.isfinite(archive.to_numpy()).all():
        raise ValueError('A finite, complete calibration archive is required.')
    return archive


def weather_scenarios(point_weather, error_vectors):
    """Add complete historical error vectors; retain the shared exact nowcast."""
    point = np.asarray(point_weather, dtype=float)
    errors = np.asarray(error_vectors, dtype=float)
    if point.ndim != 1 or errors.ndim != 2 or errors.shape[1] != len(point)-1 or not len(errors):
        raise ValueError('Use point weather (N,) and complete error vectors (S,N-1).')
    if not np.isfinite(point).all() or not np.isfinite(errors).all():
        raise ValueError('Weather inputs must be finite.')
    weather = np.tile(point, (len(errors), 1))
    weather[:, 1:] += errors
    return weather


def solve_scenario_update(controller, inputs, weather):
    """Accept a common action only after checking every scenario prediction."""
    start = time.perf_counter()
    record = rejected_plan('', origin=inputs['origin'])
    weather = np.asarray(weather, dtype=float)
    N = controller['N']; S = controller['S']; model = controller['model']
    if weather.shape != (S, N) or not np.isfinite(weather).all():
        return {**record, 'reason': 'Invalid scenario matrix.'}
    if inputs.get('current_weather_known', True) and not np.allclose(
            weather[:, 0], inputs['outdoor'][0], rtol=0, atol=1e-12):
        return {**record, 'reason': 'Scenarios must share the current nowcast.'}
    if not np.isfinite(inputs['measurement_c']):
        return {**record, 'reason': 'Missing current reading.'}
    try:
        controller['initial'].value = inputs['measurement_c']
        controller['outdoor'].value = weather
        for key in ['price', 'lower', 'upper']:
            controller[key].value = inputs[key]
        solver, options = solver_configuration()
        record['solve_attempted'] = True
        controller['problem'].solve(solver=solver, warm_start=True, **options)
    except Exception as error:
        return {**record, 'reason': str(error), 'status': 'solver_exception'}
    problem = controller['problem']
    record['status'] = problem.status
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        return {**record, 'reason': 'No accepted optimum.'}
    if problem.value is None or not np.isfinite(problem.value):
        return {**record, 'reason': 'Nonfinite objective.'}
    arrays = {}
    for key, shape in [('P', (N,)), ('T', (S, N+1)), ('s', (S, N))]:
        array = np.asarray(controller[key].value, dtype=float)
        if array.shape != shape or not np.isfinite(array).all():
            return {**record, 'reason': 'Invalid candidate '+key}
        arrays[key] = array.copy()
    power, temperature, slack = arrays['P'], arrays['T'], arrays['s']
    b = model['loss_coefficient_per_step']
    h = model['heat_coefficient_c_per_kw_step']*model['nominal_cop']
    dt = model['sampling_interval_hours']
    bill = float(dt*np.dot(inputs['price'], power))
    penalty = float(dt*controller['gamma']*slack.sum()/S)
    checks = {
        'initial_c': float(np.max(np.abs(temperature[:, 0]-inputs['measurement_c']))),
        'dynamics_c': float(np.max(np.abs(temperature[:, 1:]-(1-b)*temperature[:, :-1]
                                              -b*weather-h*power[None, :]))),
        'power_bounds_kw': max(0., -power.min(), power.max()-model['heat_pump_max_electric_kw']),
        'slack_nonnegative_c': max(0., -slack.min()),
        'lower_c': max(0., np.max(inputs['lower'][None, :]-temperature[:, 1:]-slack)),
        'upper_c': max(0., np.max(temperature[:, 1:]-inputs['upper'][None, :])),
        'objective_gbp': abs(float(problem.value)-bill-penalty)}
    if any(value > ACCEPTANCE_TOL for value in checks.values()):
        return {**record, 'reason': 'Candidate failed scenario checks.'}
    if problem.status == cp.OPTIMAL_INACCURATE:
        warnings.warn('Inaccurate optimum accepted after scenario residual checks.')
    # The session logs the mean prediction; the complete scenario arrays stay in the plan.
    return {**record, 'accepted': True, 'reason': '', 'power_kw': power,
            'temperature_c': temperature.mean(axis=0), 'slack_c': slack.mean(axis=0),
            'temperature_scenarios_c': temperature, 'slack_scenarios_c': slack,
            'outdoor_scenarios_c': weather.copy(), 'outdoor': weather.mean(axis=0),
            'price': inputs['price'].copy(), 'lower': inputs['lower'].copy(), 'upper': inputs['upper'].copy(),
            'valid_time': inputs['valid_time'].copy(), 'state_time': pd.date_range(
                inputs['origin'], periods=N+1, freq=pd.Timedelta(hours=dt)),
            'bill_gbp': bill, 'penalty_gbp': penalty, 'objective_gbp': float(problem.value),
            'checks': checks, 'solver_time_s': float(problem.solver_stats.solve_time or 0.),
            'update_wall_s': time.perf_counter()-start}


def run_weather_mpc(operation, forecasts, model, day, error_vectors, make_controller,
                    mode='scenario', seed=11, gamma=25.):
    """Run the same six steps with a mean forecast or a complete scenario set."""
    if mode not in ['mean', 'scenario']:
        raise ValueError('Select mean or scenario mode.')
    N = error_vectors.shape[1]+1
    S = 1 if mode == 'mean' else len(error_vectors)
    plant = controlled_plant(model, ErrorCase(), innovations(seed))
    session = SimulationSession(operation, model, day=day, M=48, seed=seed, plant_class=plant)
    announced = announced_schedule(operation)
    controller = make_controller(model, N=N, S=S, gamma=gamma)
    policy = FallbackPolicy(); last_command = None
    for k in range(48):
        view = session.available()                                      # 1. Read measurement.
        inputs = session.inputs(announced, forecasts, N=N)             # 2. Gather inputs.
        weather = weather_scenarios(inputs['outdoor'], error_vectors)
        if mode == 'mean':
            weather = weather.mean(axis=0, keepdims=True)
        plan = solve_scenario_update(controller, inputs, weather)       # 3. Optimise and check.
        decision = choose_command(plan, view['measurement_c'], view['lower_c'],
                                  view['upper_c'], session.capacity, policy, last_command)  # 4. Select.
        if decision['abort']:
            session.abort(decision['reason']); break
        session.advance(decision['action_kw'], plan, decision)          # 5. Apply and observe.
        last_command = session.intervals[-1]['applied_power_kw']         # 6. Repeat.
    label = 'Mean-forecast MPC' if mode == 'mean' else 'Scenario MPC'
    result = session.result(label)
    result['boundaries'] = result['boundaries'].drop(columns='fabric_temperature_c')
    result.update(scenario_count=S, scenario_seed=seed, reference='Controlled one-state')
    return result


def error_covariance(N, sigma=2., rho=.8):
    """Conditional covariance of N future AR(1) errors with innovation SD sigma."""
    if not 0 <= rho < 1 or sigma < 0:
        raise ValueError('Use 0 <= rho < 1 and a nonnegative innovation scale.')
    propagation = np.zeros((N, N))
    for j in range(N):
        propagation[j, :j+1] = rho**np.arange(j, -1, -1)
    return sigma**2*(propagation@propagation.T)


def teaching_inputs(model, measurement=20., previous_error=0., k=0, N=12, rho=.8):
    """Controller information before the next interval's weather is realised."""
    origin = pd.Timestamp('2025-02-09')+pd.Timedelta(hours=k*model['sampling_interval_hours'])
    return {'origin':origin, 'measurement_c':float(measurement),
            'outdoor':5.+rho**np.arange(1, N+1)*previous_error,
            'price':np.full(N, .30), 'lower':np.full(N, 20.), 'upper':np.full(N, 23.),
            'valid_time':pd.date_range(origin, periods=N, freq='30min'),
            'current_weather_known':False}


def monte_carlo_mpc(model, make_controller, scenario_errors, realised_errors, rho=.8, gamma=25.):
    """Evaluate two receding-horizon policies on identical independent weather paths.

    The next interval's mean weather is unknown when heating is selected.
    After the action, its weather and the new indoor temperature are observed.
    """
    R, M = realised_errors.shape; S, N = scenario_errors.shape
    controllers = {'Mean-forecast MPC':make_controller(model, N=N, S=1, gamma=gamma),
                   'Scenario MPC':make_controller(model, N=N, S=S, gamma=gamma)}
    dt = model['sampling_interval_hours']; b = model['loss_coefficient_per_step']
    h = model['heat_coefficient_c_per_kw_step']*model['nominal_cop']
    pmax = model['heat_pump_max_electric_kw']
    records = []; saved_plans = {}
    for path in range(R):
        temperatures = dict.fromkeys(controllers, 20.)
        policies = {label:FallbackPolicy() for label in controllers}
        last = dict.fromkeys(controllers, None)
        previous_error = 0.
        for k in range(M):
            for label, controller in controllers.items():
                # 1. Read state; 2. gather the conditional forecast and scenarios.
                inputs = teaching_inputs(model, temperatures[label], previous_error, k, N, rho)
                weather = inputs['outdoor'][None, :]
                if label == 'Scenario MPC':
                    weather = weather+scenario_errors
                # 3. Optimise/check; 4. select a command with the existing backup rule.
                plan = solve_scenario_update(controller, inputs, weather)
                decision = choose_command(plan, temperatures[label], 20., 23., pmax, policies[label], last[label])
                if decision['abort']:
                    raise RuntimeError('Unexpected abort in the controlled Monte Carlo case.')
                power = float(np.clip(decision['action_kw'], 0., pmax))
                # 5. Apply the command, then realise weather and the next temperature.
                actual_outdoor = 5.+realised_errors[path, k]
                next_temperature = (1-b)*temperatures[label]+b*actual_outdoor+h*power
                below = dt*max(20.-next_temperature, 0.)
                above = dt*max(next_temperature-23., 0.)
                records.append({'path':path, 'strategy':label, 'step':k,
                    'start_c':temperatures[label], 'end_c':next_temperature,
                    'forecast_current_c':inputs['outdoor'][0], 'actual_outdoor_c':actual_outdoor,
                    'power_kw':power, 'bill_gbp':dt*.30*power,
                    'below_degree_hours':below, 'above_degree_hours':above,
                    'penalty_gbp':gamma*below, 'total_cost_gbp':dt*.30*power+gamma*below,
                    'accepted':plan['accepted'], 'fallback':decision['fallback'],
                    'max_check_residual':max(plan['checks'].values()) if plan['accepted'] else np.nan})
                if path == 0 and k in [0, M//2, M-1] and plan['accepted']:
                    saved_plans[(label, k)] = plan
                temperatures[label] = next_temperature; last[label] = power
            # 6. Both controllers can use the observed weather at the next origin.
            previous_error = realised_errors[path, k]
    intervals = pd.DataFrame(records)
    metrics = intervals.groupby(['path','strategy'], sort=False).agg(
        bill_gbp=('bill_gbp','sum'), penalty_gbp=('penalty_gbp','sum'),
        total_cost_gbp=('total_cost_gbp','sum'), below_degree_hours=('below_degree_hours','sum'),
        above_degree_hours=('above_degree_hours','sum'), peak_kw=('power_kw','max'),
        accepted_solves=('accepted','sum'), fallback_count=('fallback','sum')).reset_index()
    metrics['complete'] = metrics.accepted_solves+metrics.fallback_count == M
    return metrics, intervals, saved_plans
