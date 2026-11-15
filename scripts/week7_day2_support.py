"""Supplied numerical checks for Tutorial 7.2; kept outside the teaching narrative."""
import warnings
import numpy as np
import pandas as pd
import cvxpy as cp

FEASIBILITY_TOL=1e-6
FLOW_TOL=1e-6


def solver_options(solver):
    return {"tol_gap_abs":1e-9,"tol_gap_rel":1e-9,"tol_feas":1e-9} if solver=="CLARABEL" else {}


class SiteChecks:
    """Bind checks to one shared set of physical inputs and equipment parameters."""
    def __init__(self, *, outdoor, lower, upper, base_load, import_prices, export_prices, dt, b, c, COP, P_MAX,
                 TRUE_T0, ETA_CHARGE, ETA_DISCHARGE, E_INIT, E_MIN, E_MAX,
                 BATTERY_MAX, solver):
        outdoor=np.asarray(outdoor,dtype=float).copy()
        lower=np.asarray(lower,dtype=float).copy()
        upper=np.asarray(upper,dtype=float).copy()
        base_load=np.asarray(base_load,dtype=float).copy()
        import_prices=np.asarray(import_prices,dtype=float).copy()
        export_prices=np.asarray(export_prices,dtype=float).copy()
        N=len(outdoor)
        SOLVER=solver
        SOLVER_OPTIONS=solver_options(solver)

        def physical_diagnostics(result):
            p,t=result['power_kw'],result['temperature_c']
            imp,exp=result['grid_import_kw'],result['grid_export_kw']
            ch,dis=result['charge_kw'],result['discharge_kw']
            rows=[]
            def add(check,value,unit,tolerance=FEASIBILITY_TOL):
                rows.append({'check':check,'value':float(value),'unit':unit,'tolerance':tolerance,
                             'pass':bool(value<=tolerance)})
            add('Thermal recurrence',np.max(np.abs(t[1:]-(1-b)*t[:-1]-b*outdoor-c*COP*p)),'°C')
            add('Initial temperature',abs(t[0]-TRUE_T0),'°C')
            add('Lower comfort excess',max(0.,np.max(lower-t[1:])),'°C')
            add('Upper comfort excess',max(0.,np.max(t[1:]-upper)),'°C')
            add('Heating power bounds',max(0.,-p.min(),p.max()-P_MAX),'kW')
            add('Site balance',np.max(np.abs(imp+result['pv_kw']+dis-base_load-p-ch-exp)),'kW')
            add('Grid nonnegativity',max(0.,-imp.min(),-exp.min()),'kW')
            if result['battery']:
                energy=result['energy_kwh']
                add('Battery recurrence',np.max(np.abs(energy[1:]-energy[:-1]-dt*ETA_CHARGE*ch+dt*dis/ETA_DISCHARGE)),'kWh')
                add('Initial energy',abs(energy[0]-E_INIT),'kWh')
                add('Energy bounds',max(0.,E_MIN-energy.min(),energy.max()-E_MAX),'kWh')
                add('Battery power bounds',max(0.,-ch.min(),-dis.min(),ch.max()-BATTERY_MAX,dis.max()-BATTERY_MAX),'kW')
                if result['terminal']:add('Terminal energy equality',abs(energy[-1]-energy[0]),'kWh')
            else:
                add('Absent battery powers',max(np.max(np.abs(ch)),np.max(np.abs(dis))),'kW')
            add('Charge/discharge overlap',np.sum((ch>FLOW_TOL)&(dis>FLOW_TOL)),'intervals',0)
            add('Import/export overlap',np.sum((imp>FLOW_TOL)&(exp>FLOW_TOL)),'intervals',0)
            return pd.DataFrame(rows)

        def solve_checked(problem,variables,*,pv,battery=False,terminal=True,case='',peak_weight=0.):
            record={'success':False,'status':'not_solved','case':case,'battery':battery,
                    'terminal':terminal if battery else None,'peak_weight':float(peak_weight)}
            if not problem.is_dcp():
                return {**record,'status':'not_dcp','reason':'Inspect the formulation.'}
            try:
                problem.solve(solver=SOLVER,**SOLVER_OPTIONS)
            except Exception as error:
                return {**record,'status':'solver_exception','reason':str(error)}
            record['status']=problem.status
            if problem.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:
                return {**record,'reason':'No optimal candidate; arrays were not read.'}
            expected={'power_kw':N,'temperature_c':N+1,'grid_import_kw':N,'grid_export_kw':N,
                      'charge_kw':N,'discharge_kw':N}
            if battery:expected['energy_kwh']=N+1
            arrays={}
            for key,size in expected.items():
                value=variables.get(key)
                value=value.value if hasattr(value,'value') else value
                if value is None:
                    return {**record,'reason':'Missing candidate array: '+key}
                array=np.asarray(value,dtype=float)
                if array.shape!=(size,) or not np.isfinite(array).all():
                    return {**record,'reason':'Invalid shape or nonfinite values: '+key}
                arrays[key]=array.copy()
            if not np.isfinite(problem.value):return {**record,'reason':'Nonfinite objective.'}
            arrays['pv_kw']=np.asarray(pv,dtype=float).copy()
            if arrays['pv_kw'].shape!=(N,) or not np.isfinite(arrays['pv_kw']).all():
                return {**record,'reason':'Invalid PV array.'}
            record.update(arrays)
            record['solver_objective']=float(problem.value)
            record['bill_gbp']=float(dt*np.sum(import_prices*arrays['grid_import_kw']-export_prices*arrays['grid_export_kw']))
            record['actual_peak_kw']=float(arrays['grid_import_kw'].max())
            record['peak_term_gbp']=float(peak_weight*record['actual_peak_kw'])
            record['recomputed_objective']=record['bill_gbp']+record['peak_term_gbp']
            diagnostics=physical_diagnostics(record)
            record['diagnostics']=diagnostics.to_dict(orient='records')
            record['objective_residual_gbp']=abs(record['solver_objective']-record['recomputed_objective'])
            record['success']=bool(diagnostics['pass'].all() and record['objective_residual_gbp']<=FEASIBILITY_TOL)
            if not record['success']:record['reason']='Inspect residuals, bounds, simultaneous flows and objective residual.'
            if problem.status==cp.OPTIMAL_INACCURATE:
                warnings.warn('Inaccurate status: inspect every residual before interpreting this candidate.')
            return record

        self.physical_diagnostics=physical_diagnostics
        self.solve_checked=solve_checked


def require_checked(result):
    if not result['success']:
        raise RuntimeError(f"{result['case']}: {result['status']}; {result.get('reason','check failed')}")
    return result
