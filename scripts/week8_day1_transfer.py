"""Local acceptance checks for the Tutorial 8.1 one-origin worked examples.

Accepted candidates can also be used by the optional paired simulations.
The shared heating-only solve_update and its objective checks remain unchanged.
"""
import time
import warnings
import numpy as np
import cvxpy as cp
from course_utils.week8_control import solver_configuration

TOL = 1e-6


def check_transfer_candidate(candidate, inputs, model, kind, pv=None,
                             gamma=25., lambda_peak=.75, eta=.01, lambda_tracking=1.,
                             battery=None, energy_initial=None, throughput_weight=0.):
    """Reconstruct every constraint and the objective from copied candidates."""
    if kind not in {'cost', 'peak', 'pv', 'tracking', 'pv_battery'}:
        raise ValueError('Unknown one-origin formulation.')
    n = len(inputs['outdoor'])
    p = np.asarray(candidate['power_kw'], dtype=float)
    t = np.asarray(candidate['temperature_c'], dtype=float)
    s = np.asarray(candidate['slack_c'], dtype=float)
    if any(x.shape != shape or not np.isfinite(x).all()
           for x, shape in [(p, (n,)), (t, (n+1,)), (s, (n,))]):
        raise ValueError('Invalid candidate dimensions or nonfinite values.')
    if not np.isfinite(candidate['objective_gbp']):
        raise ValueError('Nonfinite objective.')
    dt = model['sampling_interval_hours']
    b = model['loss_coefficient_per_step']
    c = model['heat_coefficient_c_per_kw_step']
    cop = model['nominal_cop']
    capacity = model['heat_pump_max_electric_kw']
    checks = {
        'initial_c': abs(t[0]-inputs['measurement_c']),
        'dynamics_c': np.max(abs(t[1:]-(1-b)*t[:-1]-b*inputs['outdoor']-c*cop*p)),
        'power_bounds_kw': max(0., -p.min(), p.max()-capacity),
        'slack_c': max(0., -s.min()),
        'lower_c': max(0., np.max(inputs['lower']-t[1:]-s)),
        'upper_c': max(0., np.max(t[1:]-inputs['upper']))}
    grid = p
    charge = discharge = np.zeros(n)
    if kind == 'pv_battery':
        if battery is None or energy_initial is None or not np.isfinite(energy_initial):
            raise ValueError('Battery settings and current stored energy are required.')
        charge = np.asarray(candidate['charge_kw'], dtype=float)
        discharge = np.asarray(candidate['discharge_kw'], dtype=float)
        energy = np.asarray(candidate['energy_kwh'], dtype=float)
        if any(x.shape != shape or not np.isfinite(x).all() for x, shape in
               [(charge, (n,)), (discharge, (n,)), (energy, (n+1,))]):
            raise ValueError('Invalid battery candidate dimensions or values.')
        checks.update(
            battery_initial_kwh=abs(energy[0]-energy_initial),
            battery_dynamics_kwh=np.max(abs(energy[1:]-energy[:-1]
                -dt*battery['eta_charge']*charge+dt*discharge/battery['eta_discharge'])),
            battery_bounds_kwh=max(0., battery['lower_kwh']-energy.min(), energy.max()-battery['upper_kwh']),
            charge_bounds_kw=max(0., -charge.min(), charge.max()-battery['power_limit_kw']),
            discharge_bounds_kw=max(0., -discharge.min(), discharge.max()-battery['power_limit_kw']),
            battery_terminal_kwh=abs(energy[-1]-battery['target_kwh']))
    if kind in {'pv', 'pv_battery'}:
        grid = np.asarray(candidate['grid_import_kw'], dtype=float)
        unused = np.asarray(candidate['unused_pv_kw'], dtype=float)
        available = np.asarray(pv, dtype=float)
        if any(x.shape != (n,) or not np.isfinite(x).all() for x in [grid, unused, available]):
            raise ValueError('Invalid PV/import arrays.')
        if np.any(available < 0):
            raise ValueError('PV availability must be nonnegative.')
        checks.update(import_nonnegative_kw=max(0., -grid.min()),
                      pv_bounds_kw=max(0., -unused.min(), np.max(unused-available)),
                      pv_balance_kw=np.max(abs(grid+available+discharge-p-charge-unused)))
    bill = float(dt*np.dot(inputs['price'], grid))
    penalty = float(gamma*dt*s.sum())
    expected = bill+penalty
    if kind == 'pv_battery':
        expected += throughput_weight*dt*float(np.sum(charge+discharge))
    if kind == 'tracking':
        target = (inputs['lower']+inputs['upper'])/2
        expected += lambda_tracking*dt*float(np.sum((t[1:]-target)**2))
    if kind == 'peak':
        peak = float(candidate['peak_epigraph_kw'])
        if not np.isfinite(peak):
            raise ValueError('Nonfinite peak epigraph.')
        checks['peak_epigraph_kw'] = max(0., -peak, np.max(p-peak))
        expected = lambda_peak*peak+penalty+eta*bill
    checks['objective_gbp'] = abs(candidate['objective_gbp']-expected)
    return {k: float(v) for k, v in checks.items()}, bill, penalty


def solve_transfer(problem, variables, inputs, model, kind, pv=None,
                   gamma=25., lambda_peak=.75, eta=.01, lambda_tracking=1.,
                   battery=None, energy_initial=None, throughput_weight=0.):
    """Solve, copy and check a cost/peak/PV plan; retain rejection reasons."""
    start = time.perf_counter()
    record = dict(kind=kind, origin=inputs['origin'], accepted=False,
                  status='not_solved', reason='', solve_attempted=False)
    try:
        n = len(inputs['outdoor'])
        if n < 1 or not np.isfinite(inputs['measurement_c']):
            raise ValueError('Missing initial observation or empty horizon.')
        for name in ['outdoor', 'price', 'lower', 'upper']:
            x = np.asarray(inputs[name], dtype=float)
            if x.shape != (n,) or not np.isfinite(x).all():
                raise ValueError('Invalid horizon input: '+name)
        if len(inputs['valid_time']) != n or np.any(inputs['lower'] > inputs['upper']):
            raise ValueError('Invalid horizon times or bounds.')
        if not problem.is_dcp():
            raise ValueError('Problem is not DCP.')
        solver, options = solver_configuration()
        record['solve_attempted'] = True
        problem.solve(solver=solver, **options)
        record['status'] = problem.status
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            raise ValueError('No optimal candidate.')
        candidate = {key: np.asarray(var.value, dtype=float).copy()
                     for key, var in variables.items()}
        candidate['objective_gbp'] = float(problem.value)
        checks, bill, penalty = check_transfer_candidate(
            candidate, inputs, model, kind, pv, gamma, lambda_peak, eta, lambda_tracking,
            battery, energy_initial, throughput_weight)
        record['checks'] = checks
        if any(v > TOL for v in checks.values()):
            raise ValueError('Candidate failed numerical acceptance checks.')
        if problem.status == cp.OPTIMAL_INACCURATE:
            warnings.warn('Inaccurate candidate accepted after local residual checks.')
        record.update(candidate, accepted=True, bill_gbp=bill, penalty_gbp=penalty,
                      valid_time=inputs['valid_time'].copy())
    except Exception as error:
        record['reason'] = str(error)
    record['update_wall_s'] = time.perf_counter()-start
    return record
