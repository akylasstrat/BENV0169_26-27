"""Simulation and plotting support for independent Tutorial 8.1 activities.

Each notebook activity supplies its own objective and constraints. This module
only manages independent plants, checked actions, logging and realised metrics.
"""
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from course_utils.week8_control import SimulationSession, announced_schedule, realised_metrics


def daylight_pv(times):
    """Illustrative, perfectly forecast PV availability, electrical kW."""
    times = pd.DatetimeIndex(times)
    hour = times.hour.to_numpy()+times.minute.to_numpy()/60
    return 2*np.maximum(np.sin(np.pi*(hour-6)/12), 0.)


def run_case(planner, operation, forecasts, model, *, day, seed, M=48,
             N=48, fixed=False, horizon=None, pv_function=None):
    """Same first-action loop as Task D; stop explicitly on a rejected plan."""
    session = SimulationSession(operation, model, day, M=M, seed=seed)
    schedule = announced_schedule(operation)
    plans, horizons = [], []
    first_plan = None
    for k in range(M):
        view = session.available()
        n = int(horizon(k)) if horizon else N
        if fixed and k == 0 and n < M:
            raise ValueError('The fixed initial plan must cover all reporting intervals.')
        if not fixed or k == 0:
            inputs = session.inputs(schedule, forecasts, n)
            plan = planner(inputs)
            if not plan['accepted']:
                session.abort('Optional example stopped: '+plan.get('reason', plan['status']))
                break
            # Copies retain what was known and solved at this origin.
            plan = copy.deepcopy(plan)
            plan.update({key: copy.deepcopy(inputs[key]) for key in
                         ['outdoor', 'price', 'lower', 'upper']})
            plan['state_time'] = pd.date_range(view['origin'], periods=n+1,
                                              freq=pd.Timedelta(hours=session.dt))
            plans.append(plan)
            if k == 0:
                first_plan = plan
        else:
            plan = first_plan
        index = k if fixed else 0
        # No plant transition takes place unless a copied plan was accepted.
        session.advance(float(plan['power_kw'][index]), plan,
                        update_wall_s=plan.get('update_wall_s', 0.) if index == 0 else 0.,
                        prediction_index=index)
        horizons.append({'interval': k, 'origin': view['origin'], 'N': n,
                         'endpoint': view['origin']+pd.Timedelta(hours=n*session.dt)})
    result = session.result('Fixed plan' if fixed else 'MPC',
                            fixed_plan_failed=fixed and first_plan is None)
    result['all_plans'] = plans
    result['horizons'] = pd.DataFrame(horizons)
    rows = result['intervals']
    if len(rows):
        available = (pv_function(rows.interval_start) if pv_function else np.zeros(len(rows)))
        rows['realised_pv_kw'] = available
        rows['grid_import_kw'] = np.maximum(rows.applied_power_kw.to_numpy()-available, 0.)
        rows['interval_grid_cost_gbp'] = session.dt*rows.price_gbp_per_kwh*rows.grid_import_kw
    return result


def run_pair(planner, operation, forecasts, model, *, day, seed, pv_function=None):
    return {label: run_case(planner, operation, forecasts, model, day=day,
                           seed=seed, fixed=fixed, pv_function=pv_function)
            for label, fixed in [('Fixed plan', True), ('MPC', False)]}


def case_metrics(runs):
    records = []
    for label, run in runs.items():
        score = realised_metrics(run)
        rows = run['intervals']
        ok = run['complete'] and not run['fixed_plan_failed']
        midpoint = (rows.lower_c+rows.upper_c)/2 if ok else None
        records.append({'Policy': label, 'Complete': ok,
                        'Heating (kWh)': score['energy_kwh'],
                        'Grid bill (GBP)': rows.interval_grid_cost_gbp.sum() if ok else np.nan,
                        'Heating peak (kW)': score['peak_kw'],
                        'Below band (C h)': score['below_degree_hours'],
                        'Above band (C h)': score['above_degree_hours'],
                        'Midpoint RMSE (C)': np.sqrt(np.mean((rows.true_end_c-midpoint)**2)) if ok else np.nan})
    return pd.DataFrame(records)


def record_run(session, plans, horizons):
    """Package already-applied actions for figures; does not solve or advance."""
    result = session.result('Optional controller')
    result['all_plans'] = plans
    result['horizons'] = pd.DataFrame(horizons)
    rows = result['intervals']
    if len(rows) and 'grid_import_kw' not in rows:
        rows['grid_import_kw'] = rows.applied_power_kw
        rows['interval_grid_cost_gbp'] = session.dt*rows.price_gbp_per_kwh*rows.grid_import_kw
    return result


def battery_metrics(runs):
    """Include end energy and battery losses when interpreting grid bills."""
    table = case_metrics(runs)
    for index, run in enumerate(runs.values()):
        rows = run['intervals']
        ok = run['complete'] and len(rows) > 0
        values = {
            'Grid import (kWh)': run['dt']*rows.grid_import_kw.sum() if ok else np.nan,
            'PV curtailed (kWh)': run['dt']*rows.curtailed_pv_kw.sum() if ok else np.nan,
            'Initial battery (kWh)': rows.battery_start_kwh.iloc[0] if ok else np.nan,
            'Final battery (kWh)': rows.battery_end_kwh.iloc[-1] if ok else np.nan,
            'Battery losses (kWh)': (run['dt']*(rows.charge_kw-rows.discharge_kw).sum()
                -rows.battery_end_kwh.iloc[-1]+rows.battery_start_kwh.iloc[0]) if ok else np.nan,
            'Max simultaneous flow (kW)': np.minimum(rows.charge_kw, rows.discharge_kw).max() if ok else np.nan}
        for name, value in values.items():
            table.loc[index, name] = value
    return table


def plot_battery(runs):
    fig, axes = plt.subplots(4, 1, figsize=(12.5, 10), sharex=True)
    for (label, run), colour in zip(runs.items(), ['tab:green', 'tab:blue']):
        rows = run['intervals']
        times = pd.DatetimeIndex(run['boundaries'].state_time)
        axes[0].stairs(rows.grid_import_kw, mdates.date2num(times), baseline=None, label=label, color=colour)
        axes[1].stairs(rows.charge_kw-rows.discharge_kw, mdates.date2num(times), baseline=None, label=label, color=colour)
        axes[2].plot(times, np.r_[rows.battery_start_kwh.iloc[0], rows.battery_end_kwh], color=colour, label=label)
        axes[3].stairs(rows.used_pv_kw, mdates.date2num(times), baseline=None, label=label+' used PV', color=colour)
    axes[2].axhline(.5, color='0.4', linestyle='--', label='Energy bounds')
    axes[2].axhline(9.5, color='0.4', linestyle='--')
    axes[3].stairs(rows.realised_pv_kw, mdates.date2num(times), baseline=None,
                   color='tab:orange', linestyle=':', label='Available PV')
    for ax, label in zip(axes, ['Grid import (kW)', 'Charge - discharge (kW)', 'Stored energy (kWh)', 'PV (kW)']):
        ax.set_ylabel(label)
        ax.legend(loc='upper left', ncol=3)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,4)))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel('Time on 3 February 2025')
    fig.suptitle('Optional B: realised grid, battery and controllable PV operation')
    fig.tight_layout()
    return fig


def plot_pair(runs, title, *, pv=False):
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8), sharex=True)
    for (label, run), colour in zip(runs.items(), ['tab:green', 'tab:blue']):
        rows, states = run['intervals'], run['boundaries']
        if not len(rows):
            continue
        times = pd.DatetimeIndex(states.state_time)
        axes[0].plot(times, states.true_temperature_c, label=label, color=colour)
        axes[1].stairs(rows.applied_power_kw, mdates.date2num(times), label=label, color=colour)
        axes[2].plot(times, np.r_[0., rows.interval_grid_cost_gbp.cumsum()], label=label, color=colour)
    reference = next(iter(runs.values()))['intervals']
    if len(reference):
        ends = pd.DatetimeIndex(reference.interval_end)
        axes[0].plot(ends, reference.lower_c, '--', color='0.4', label='Comfort bounds')
        axes[0].plot(ends, reference.upper_c, '--', color='0.4')
        if pv:
            axes[1].step(reference.interval_start, reference.realised_pv_kw, where='post',
                         color='tab:orange', linestyle='--', label='Available PV')
    for i, ax in enumerate(axes):
        ax.legend(loc='upper right' if i == 1 and not pv else 'upper left', ncol=3)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Indoor temperature (°C)')
    axes[1].set_ylabel('Electrical power (kW)')
    axes[2].set_ylabel('Cumulative grid bill (GBP)')
    axes[2].xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,4)))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[2].set_xlabel('Time on 3 February 2025')
    fig.suptitle(title+' | realised operation | same objective in both policies')
    fig.tight_layout()
    return fig


def export_case(runs, metrics, fig, name, output_dir, figure_dir, enabled=True):
    if not enabled:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir/(name+'_metrics.csv'), index=False)
    for label, run in runs.items():
        prefix = name+'_'+label.lower().replace(' ', '_')
        for key in ['intervals', 'boundaries', 'horizons', 'forecast_records']:
            run[key].to_csv(output_dir/(prefix+'_'+key+'.csv'), index=False)
        pd.DataFrame([{'origin': p['origin'], 'status': p['status'],
                       'accepted': p['accepted'], 'max_residual': max(p['checks'].values()),
                       'objective': p['objective_gbp'], 'N': len(p['power_kw'])}
                      for p in run['all_plans']]).to_csv(output_dir/(prefix+'_solves.csv'), index=False)
    for ext in ['png', 'svg']:
        fig.savefig(figure_dir/(name+'.'+ext), dpi=160)
