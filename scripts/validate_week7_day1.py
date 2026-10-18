"""Private fresh-kernel, independent LP, physical and export checks for 7.1."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from build_week7_day1 import ROOT, STEM, FIGURES, TASKS, SLIDE_MINUTES, make_notebook

OUTPUT=ROOT/'teaching/week7_day1_outputs'
CHECKS=r'''
from scipy.optimize import linprog
from unittest.mock import patch
from types import SimpleNamespace
import importlib.metadata as _metadata
import sys as _sys
import hashlib as _hashlib

# Independent half-plane intersections and vertex arithmetic, not the plotting grid.
_A=np.array([[1,1],[0,1],[2,1],[-1,0],[0,-1]],dtype=float)
_rhs=np.array([4,3,6,0,0],dtype=float)
_vertices=[]
for _i in range(5):
    for _j in range(_i+1,5):
        _matrix=_A[[_i,_j]]
        if abs(np.linalg.det(_matrix))<1e-12:continue
        _point=np.linalg.solve(_matrix,_rhs[[_i,_j]])
        if np.all(_A@_point<=_rhs+1e-12) and not any(np.allclose(_point,v) for v in _vertices):
            _vertices.append(_point)
_expected=np.array([[0,0],[3,0],[2,2],[1,3],[0,3]],dtype=float)
assert len(_vertices)==5
assert all(any(np.allclose(v,p,atol=1e-12) for p in _vertices) for v in _expected)
np.testing.assert_array_equal(_expected@np.array([1,-2]),[0,3,-2,-5,-6])
_toy=linprog([1,-2],A_ub=_A[:3],b_ub=_rhs[:3],bounds=[(0,None),(0,None)],method='highs')
assert _toy.success
np.testing.assert_allclose(_toy.x,[0,3],atol=1e-12)
np.testing.assert_allclose(_toy.fun,-6,atol=1e-12)
np.testing.assert_allclose(best_xy,_toy.x,atol=1e-12)
np.testing.assert_allclose(best_objective,_toy.fun,atol=1e-12)
np.testing.assert_allclose(graph_solution,_toy.x,atol=FEASIBILITY_TOL)
_graph_residuals=_A@graph_solution-_rhs
active_labels=np.array(['x+y=4','y=3','2x+y=6','x=0','y=0'])[np.abs(_graph_residuals)<=FEASIBILITY_TOL]
assert set(active_labels)=={'x=0','y=3'}
np.testing.assert_array_equal(feasible,(X+Y<=4)&(Y<=3)&(2*X+Y<=6)&(X>=0)&(Y>=0))
np.testing.assert_array_equal(objective_grid,X-2*Y)


# Detailed implementation diagnostics stay private, outside the student notebook.
fixture_power=cp.Variable(nonneg=True)
fixture_next=cp.Variable()
fixture_reachable_max=(1-b)*TRUE_T0+b*outdoor[0]+c*COP*P_MAX
fixture_lower=fixture_reachable_max+1.
fixture_problem=cp.Problem(cp.Minimize(fixture_power),[
    fixture_power<=P_MAX,fixture_next==(1-b)*TRUE_T0+b*outdoor[0]+c*COP*fixture_power,
    fixture_next>=fixture_lower])
fixture_problem.solve(solver=SOLVER,**SOLVER_OPTIONS)
fixture_record={'status':fixture_problem.status}
def private_diagnostics(result):
    p,t,s=result['power_kw'],result['temperature_c'],result['slack_c']
    return {'max_dynamic_residual_c':float(np.abs(t[1:]-((1-b)*t[:-1]+b*outdoor+c*COP*p)).max()),
        'initial_residual_c':float(abs(t[0]-result['initial_c'])),
        'power_bound_excess_kw':float(max(0.,-p.min(),p.max()-P_MAX)),
        'lower_constraint_excess_c':float(max(0.,np.max(lower-t[1:]-s))),
        'upper_constraint_excess_c':float(max(0.,np.max(t[1:]-upper))),
        'slack_nonnegative_excess_c':float(max(0.,-s.min())),
        'objective_residual':abs(result['solver_objective']-result['recomputed_objective'])}

assert N==48 and len(state_time)==49 and dt==.5
assert interval_start[0]==pd.Timestamp('2025-02-03') and state_time[-1]==pd.Timestamp('2025-02-04')
assert intervals.interval_start.tolist()==list(interval_start)
assert intervals.interval_end.tolist()==list(state_time[1:])
assert (intervals.interval_end-intervals.interval_start).eq(pd.Timedelta(minutes=30)).all()
assert hard_problem.is_dcp() and soft_problem.is_dcp() and peak_problem.is_dcp()
assert hard_problem.is_qp() and len(hard_problem.variables())==2
assert all(v is not slack and v is not P_peak for v in hard_problem.variables())
assert len(hard_constraints)==len(base_constraints)+2
assert len(soft_constraints)==len(base_constraints)+2
assert all(a is b for a,b in zip(soft_constraints[:len(base_constraints)],base_constraints))
assert peak_problem.objective.args[0] is P_peak
assert fixture_record['status'] in {cp.INFEASIBLE,cp.INFEASIBLE_INACCURATE}
assert fixture_lower>fixture_reachable_max
np.testing.assert_allclose(hard_bill_check,hard_solution['bill_gbp'],atol=1e-12)
np.testing.assert_allclose(hard_energy_check,hard_metrics['energy_kwh'],atol=1e-12)

_all=[hard_solution,flat_solution,soft_solution,peak_solution,*penalty_solutions.values(),*error_solutions.values()]

for _result in _all:
    if _result['success']:_result['diagnostics']=private_diagnostics(_result)
hard_metrics.update(hard_solution['diagnostics'])
matched_states={'Hard':simulate_temperature(hard_solution['power_kw'],TRUE_T0)}
matched_table=pd.DataFrame([{'case':'Hard',
    'max_replay_difference_c':float(np.abs(matched_states['Hard']-hard_solution['temperature_c']).max()),
    'propagated_residual_bound_c':N*hard_metrics['max_dynamic_residual_c']+1e-10}])

_successes=[r for r in _all if r['success']]
for _r in _successes:
    _p,_t,_s=_r['power_kw'],_r['temperature_c'],_r['slack_c']
    assert (_p.shape,_t.shape,_s.shape)==((48,),(49,),(48,))
    assert np.isfinite(np.r_[_p,_t,_s]).all()
    _oracle=np.empty(49);_oracle[0]=_r['initial_c']
    for _k in range(48):
        _oracle[_k+1]=_oracle[_k]+b*(outdoor[_k]-_oracle[_k])+c*COP*_p[_k]
    np.testing.assert_allclose(_t,_oracle,rtol=0,atol=FEASIBILITY_TOL)
    assert _p.min()>=-FEASIBILITY_TOL and _p.max()<=P_MAX+FEASIBILITY_TOL
    assert np.max(lower-_t[1:]-_s)<=FEASIBILITY_TOL
    assert np.max(_t[1:]-upper)<=FEASIBILITY_TOL
    assert _s.min()>=-FEASIBILITY_TOL
    if _r['hard']:np.testing.assert_array_equal(_s,np.zeros(N))
    _bill=sum(float(q)*float(p) for q,p in zip(_p,_r['prices']))*dt
    _penalty=sum(float(s) for s in _s)*dt*_r['gamma']
    np.testing.assert_allclose([_r['bill_gbp'],_r['weighted_slack_gbp']],[_bill,_penalty],rtol=0,atol=1e-10)
    _obj=float(_p.max()) if _r['objective_kind']=='peak' else _bill+_penalty
    np.testing.assert_allclose(_r['recomputed_objective'],_obj,rtol=0,atol=1e-10)
    assert abs(_r['solver_objective']-_obj)<=FEASIBILITY_TOL
    assert not np.shares_memory(_p,power.value)
    assert not np.shares_memory(_r['prices'],price.value)
    assert max(_r['diagnostics'].values())<=FEASIBILITY_TOL

# Independent condensed LP: eliminate all temperature variables by recurrence.
_zero=simulate_temperature(np.zeros(N))
_H=np.column_stack([simulate_temperature(np.eye(N)[j])[1:]-_zero[1:] for j in range(N)])
_lp_rhs=np.r_[upper-_zero[1:],_zero[1:]-lower]
_lp_A=np.vstack([_H,-_H])
_heating_oracles={}
for _name,_r in tariff_solutions.items():
    _lp=linprog(_r['prices']*dt,A_ub=_lp_A,b_ub=_lp_rhs,bounds=[(0,P_MAX)]*N,method='highs')
    assert _lp.success
    np.testing.assert_allclose(_r['bill_gbp'],_lp.fun,rtol=0,atol=FEASIBILITY_TOL)
    _heating_oracles[_name]=float(_lp.fun)
_peak_A=np.vstack([np.c_[_lp_A,np.zeros(2*N)],np.c_[np.eye(N),-np.ones(N)]])
_peak_lp=linprog(np.r_[np.zeros(N),1.],A_ub=_peak_A,b_ub=np.r_[_lp_rhs,np.zeros(N)],
    bounds=[(0,P_MAX)]*N+[(0,None)],method='highs')
assert _peak_lp.success
np.testing.assert_allclose(peak_solution['solver_objective'],_peak_lp.fun,rtol=0,atol=FEASIBILITY_TOL)
for _weight,_r in penalty_solutions.items():
    assert _r['success']
    _soft_A=np.vstack([np.c_[_H,np.zeros((N,N))],np.c_[-_H,-np.eye(N)]])
    _soft_lp=linprog(np.r_[tou_prices*dt,np.full(N,_weight*dt)],
        A_ub=_soft_A,b_ub=_lp_rhs,bounds=[(0,P_MAX)]*N+[(0,None)]*N,method='highs')
    assert _soft_lp.success
    np.testing.assert_allclose(_r['solver_objective'],_soft_lp.fun,rtol=0,atol=FEASIBILITY_TOL)
    _shortfall=np.maximum(lower-_r['temperature_c'][1:],0.)
    np.testing.assert_allclose(_r['slack_c'],_shortfall,rtol=0,atol=FEASIBILITY_TOL)

# Final action influences the final boundary only; c already includes dt.
_last=np.zeros(N);_last[-1]=1.
_distinct=simulate_temperature(_last)
np.testing.assert_array_equal(_distinct[:-1],_zero[:-1])
np.testing.assert_allclose(_distinct[-1]-_zero[-1],c*COP,rtol=0,atol=1e-12)
assert abs((_distinct[-1]-_zero[-1])-c*COP*dt)>1e-3
for _name,_replay in matched_states.items():
    _row=matched_table.set_index('case').loc[_name]
    assert _row.max_replay_difference_c<=_row.propagated_residual_bound_c
    assert _row.max_replay_difference_c<=FEASIBILITY_TOL and _replay[0]==TRUE_T0
_hand=deviation_metrics(np.array([1.,0.,.5,0.]),np.array([0.,1.,0.,.25]))
np.testing.assert_allclose([_hand['hours_below_band'],_hand['hours_above_band'],_hand['below_band_degree_hours'],
    _hand['above_band_degree_hours'],_hand['total_degree_hours'],_hand['max_outside_band_c']],[1,1,.75,.625,1.375,1])
assert all(v==0 for v in deviation_metrics(np.zeros(4),np.zeros(4)).values())
_tiny=deviation_metrics(np.array([1e-8]),np.zeros(1))
assert _tiny['hours_below_band']==0 and _tiny['below_band_degree_hours']==5e-9
np.testing.assert_allclose([.4*dt,2.5*.4*dt],[.2,.5],atol=1e-15)
_shortfalls=[penalty_solutions[w]['slack_c'].sum()*dt for w in PENALTY_WEIGHTS]
assert len(penalty_solutions)==len(PENALTY_WEIGHTS) and set([.25,2.5,25.])<=set(PENALTY_WEIGHTS)
assert all(a>=b-FEASIBILITY_TOL for a,b in zip(_shortfalls,_shortfalls[1:]))

# Price Parameter reuse and fair cross-evaluation, without schedule-equality tests.
assert set(cross_costs.index)==set(cross_costs.columns)=={'Flat','ToU'}
for _plan,_r in tariff_solutions.items():
    for _evaluation,_prices in tariffs.items():
        np.testing.assert_allclose(cross_costs.loc[_plan,_evaluation],np.dot(_r['power_kw'],_prices)*dt,atol=1e-12)
    assert _r['initial_c']==TRUE_T0 and _r['hard']
np.testing.assert_array_equal(flat_prices,np.full(N,tou_prices.mean()))
np.testing.assert_array_equal(tou_prices,conditions.import_price_gbp_per_kwh.to_numpy())
assert cross_costs.loc['ToU','ToU']<=cross_costs.loc['Flat','ToU']+FEASIBILITY_TOL
assert cross_costs.loc['Flat','Flat']<=cross_costs.loc['ToU','Flat']+FEASIBILITY_TOL
np.testing.assert_allclose(scheduling_value_tou,cross_costs.loc['Flat','ToU']-cross_costs.loc['ToU','ToU'],atol=1e-12)
assert scheduling_value_tou>=-FEASIBILITY_TOL
assert peak_solution['power_kw'].max()<=hard_solution['power_kw'].max()+FEASIBILITY_TOL
assert peak_solution['objective_units']=='kW' and 'total_objective_gbp' not in peak_solution
assert hard_solution['objective_units']==soft_solution['objective_units']=='GBP'
assert abs(peak_solution['solver_objective']-peak_solution['bill_gbp'])>1e-3

# Initial observation varies; true replay origin and fixed schedule do not.
assert len(error_table)==10 and set(error_solutions)=={(m,e) for m in ['Hard','Soft'] for e in [-1.,-.5,0.,.5,1.]}
assert error_table.true_initial_c.eq(20.).all()
np.testing.assert_allclose(error_table.measured_initial_c,20+error_table.error_c,atol=0)
for _row in error_table.itertuples():
    _r=error_solutions[_row.mode,_row.error_c]
    if not _r['success']:
        assert error_replays[_row.mode,_row.error_c] is None
        continue
    _realised=error_replays[_row.mode,_row.error_c]
    assert _realised.shape==(49,) and _realised[0]==20.
    assert _r['initial_c']==20+_row.error_c
    np.testing.assert_array_equal(_r['prices'],tou_prices)
    np.testing.assert_allclose(_r['temperature_c']-_realised,_row.error_c*(1-b)**np.arange(49),rtol=0,atol=FEASIBILITY_TOL)
    _low=np.array([max(float(L)-float(T),0.) for L,T in zip(lower,_realised[1:])])
    _high=np.array([max(float(T)-float(U),0.) for U,T in zip(upper,_realised[1:])])
    for _key,_value in deviation_metrics(_low,_high).items():
        np.testing.assert_allclose(getattr(_row,_key),_value,atol=1e-12)
assert error_solutions['Hard',-1.]['success']  # First endpoint remains reachable.

# Reject stale, failed, non-finite, corrupted and wrongly interpreted objectives.
_saved_power=hard_solution['power_kw'].copy()
initial_temperature.value=TRUE_T0;price.value=tou_prices.copy();gamma.value=25.
with patch.object(hard_problem,'solve',side_effect=cp.error.SolverError('deliberate private fixture')):
    _failure=solve_case(hard_problem)
assert not _failure['success'] and _failure['status']=='solver_error' and 'power_kw' not in _failure
assert all(v.value is None for v in hard_problem.variables())
np.testing.assert_array_equal(hard_solution['power_kw'],_saved_power)
_bad=cp.Problem(objective,hard_constraints+[temperature[1]>=100.])
_failed_infeasible=solve_case(_bad)
assert not _failed_infeasible['success'] and _failed_infeasible['status'] in {cp.INFEASIBLE,cp.INFEASIBLE_INACCURATE}
assert 'power_kw' not in _failed_infeasible
_bad_upper=cp.Problem(soft_objective,soft_constraints+[temperature[1]>=upper[0]+1.])
assert not solve_case(_bad_upper,hard=False,objective_kind='soft',slack_variable=slack,penalty_rate=float(gamma.value))['success']
_restored=solve_case(hard_problem);assert _restored['success']
with warnings.catch_warnings(record=True) as _warnings:
    warnings.simplefilter('always')
    _inaccurate=snapshot_result(SimpleNamespace(status=cp.OPTIMAL_INACCURATE,value=hard_problem.value,constraints=hard_problem.constraints))
assert _inaccurate['success'] and len(_warnings)>=1
assert not snapshot_result(SimpleNamespace(status=cp.OPTIMAL,value=float('nan'),constraints=hard_problem.constraints))['success']
power.value=np.full(N,P_MAX+1.)
_corrupted=snapshot_result(SimpleNamespace(status=cp.OPTIMAL,value=hard_problem.value,constraints=hard_problem.constraints))
assert not _corrupted['success'] and power.value.max()>=P_MAX+1.
try:schedule_metrics(_corrupted)
except RuntimeError:pass
else:raise AssertionError('Rejected raw values entered successful metrics')
_restored_peak=solve_case(peak_problem,objective_kind='peak',peak_variable=P_peak)
assert _restored_peak['success']
assert not snapshot_result(peak_problem,objective_kind='bill')['success']
assert solve_case(hard_problem)['success']
np.testing.assert_array_equal(hard_solution['power_kw'],_saved_power)

# All six task cells stop intentionally; the optional extension has no TODO.
_student=nbformat.read(ROOT/'tutorial_7_1_building_optimisation.ipynb',4)
for _cell in _student.cells:
    if 'task' not in _cell.metadata.get('tags',[]):continue
    _letter=_cell.metadata.tags[-1]
    try:exec(_cell.source,dict(globals()))
    except NotImplementedError as _error:assert 'Task '+_letter in str(_error)
    else:raise AssertionError('Student task did not stop: '+_letter)


# The student-facing tables report the same common quantities for each case.
np.testing.assert_allclose(hard_summary.iloc[0].mean_temperature_c,hard_temperature[1:].mean(),atol=1e-12)
np.testing.assert_allclose(electricity_bill_gbp,hard_solution['bill_gbp'],atol=1e-12)
np.testing.assert_allclose(total_energy_kwh,hard_metrics['energy_kwh'],atol=1e-12)
np.testing.assert_allclose(rebuilt_bill,flat_solution['bill_gbp'],rtol=0,atol=FEASIBILITY_TOL)
assert all(np.max(np.abs(c.violation()))<=FEASIBILITY_TOL for c in rebuilt_constraints)
for _frame in [hard_summary,tariff_summary,peak_summary]:
    assert set(SUMMARY_COLUMNS)<=set(_frame.columns)
for _case in _successes:
    _summary=schedule_metrics(_case)
    np.testing.assert_allclose(_summary['mean_temperature_c'],_case['temperature_c'][1:].mean(),atol=1e-12)
    np.testing.assert_allclose(_summary['peak_interval_energy_kwh'],_case['power_kw'].max()*dt,atol=1e-12)

validation_payload={'status':'PASS','python':_sys.version,'executable':_sys.executable,
    'packages':{p:_metadata.version(p) for p in ['numpy','pandas','cvxpy','clarabel','scipy','matplotlib','nbformat','nbclient','ipykernel']},
    'solver':SOLVER,'solver_options':SOLVER_OPTIONS,'installed_solvers':available_solvers,
    'N':N,'N_states':len(state_time),'period_start':str(interval_start[0]),'period_end':str(interval_end[-1]),
    'feasibility_tolerance':FEASIBILITY_TOL,'comfort_count_tolerance_c':COMFORT_COUNT_TOL,
    'graphical_lp_oracle':{'vertices':_expected.tolist(),'objective_values':(_expected@np.array([1,-2])).tolist(),'optimum':_toy.x.tolist(),'objective':float(_toy.fun)},
    'condensed_hard_lp_oracles_gbp':_heating_oracles,'peak_lp_oracle_kw':float(_peak_lp.fun),
    'hard_soft_metrics':comfort_comparison.to_dict(orient='records'),
    'penalty_metrics':penalty_table.to_dict(orient='records'),
    'tariff_cross_costs':cross_costs.to_dict(),'tariff_energy':tariff_energy.to_dict(orient='records'),
    'scheduling_value_tou_gbp':scheduling_value_tou,'peak_metrics':peak_metrics.to_dict(orient='records'),
    'initial_error_metrics':error_table.to_dict(orient='records'),
    'matched_max_difference_c':float(matched_table.max_replay_difference_c.max()),
    'fixture_status':fixture_record['status'],'forced_solver_exception':'PASS: no stale arrays',
    'failed_experiment_cases':[{'status':r['status'],'reason':r.get('reason','')} for r in _all if not r['success']],
    'snapshot_sha256':_hashlib.sha256(b''.join(r['power_kw'].tobytes()+r['temperature_c'].tobytes() for r in _successes)).hexdigest(),
    'checks':'PASS: graphical vertices and LP oracle; condensed hard/soft/peak LP oracles; N+1 endpoints; cCOP units; physical residuals; objective units; upper-hard soft comfort; tariff cross-evaluation and optimality; fixed replay truth; stale/failure/inaccurate fixtures'}
print(validation_payload['checks'])

'''


def static_checks():
    pair=[nbformat.read(ROOT/(STEM+suffix+'.ipynb'),4) for suffix in ['', '_solution']]
    for nb,solved in zip(pair,[False,True]):
        nbformat.validate(nb);expected=make_notebook(solved)
        assert [(c.id,c.source) for c in nb.cells]==[(c.id,c.source) for c in expected.cells]
        for c in nb.cells:
            if c.cell_type=='code':compile(c.source,c.id,'exec')
    assert len(pair[0].cells)==len(pair[1].cells)
    letters=[]
    for a,b in zip(pair[0].cells,pair[1].cells):
        assert (a.id,a.cell_type,a.metadata.get('tags'))==(b.id,b.cell_type,b.metadata.get('tags'))
        if a.cell_type=='code':assert not a.outputs and a.execution_count is None
        if 'task' in a.metadata.get('tags',[]):
            letters.append(a.metadata.tags[-1]);assert 'NotImplementedError' in a.source and 'NotImplementedError' not in b.source
        else:assert a.source==b.source
    assert letters==list('ABCDEF')
    cells={c.id:c for c in pair[0].cells};ids=list(cells)
    assert ids.index('w71-task-a')<ids.index('w71-toy-figure')<ids.index('w71-graph-solver')
    assert ids.index('w71-inputs-text')>ids.index('w71-graph-solver')
    assert ids.index('w71-task-b')<ids.index('w71-break')<ids.index('w71-task-c')
    assert ids.index('w71-result-checks')>ids.index('w71-task-c')
    assert ids.index('w71-soft-parameters')>ids.index('w71-task-d')
    assert 'slack=cp.Variable' not in '\n'.join(c.source for c in pair[0].cells[:ids.index('w71-soft-parameters')])
    assert 'slack' not in next(c for c in pair[1].cells if c.id=='w71-task-b').source
    assert ids.index('w71-task-f')<ids.index('w71-initial-error-case')<ids.index('w71-error-sweep')
    assert 'NotImplementedError' not in cells['w71-initial-error-case'].source
    # Literal solutions must not occur in supplied starter text or tests.
    early='\n'.join(c.source for c in pair[0].cells[:ids.index('w71-task-a')+1])
    for answer in ['(0,3)','(0, 3)','best_objective=-6','best_xy=np.array([0','(4,2)','bill_gbp']:
        assert answer not in early
    all_text='\n'.join(c.source for c in pair[0].cells)
    for stale in ['C:/','C:\\','two-interval allocation','slide 33','slide 12; 12 min','64 minutes','slides 25–27']:
        assert stale not in all_text,stale
    assert 'course_utils.control' not in all_text and 'course_utils.mpc' not in all_text
    assert [row[1] for row in TASKS]==[11,21,24,28,36,40]
    assert sum(row[2] for row in TASKS)==62
    assert len(SLIDE_MINUTES)==42 and sum(SLIDE_MINUTES)==200
    assert [sum(SLIDE_MINUTES[a:b]) for a,b in [(0,11),(11,21),(21,31),(31,42)]]==[50]*4
    assert all(SLIDE_MINUTES[slide-1]==minutes for _,slide,minutes in TASKS)
    return pair


def execute(nb,kernel_root,result_path):
    executed=copy.deepcopy(nb)
    checks='import nbformat\n'+CHECKS+'\nPath('+repr(str(result_path))+').write_text(json.dumps(validation_payload,indent=2),encoding="utf8")'
    executed.cells.append(nbformat.v4.new_code_cell(checks))
    manager=KernelManager(kernel_name='week7-day1-validation',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client=NotebookClient(executed,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}})
    try:client.execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    executed.cells.pop()
    return executed


def output_checks(snapshot):
    import pandas as pd
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week7_day1_figure_manifest.csv',dtype={'slide':str})
    assert manifest.id.tolist()==[row[0] for row in FIGURES] and manifest.id.is_unique
    assert manifest.slide.tolist()==[str(row[1]) for row in FIGURES]
    for row in manifest.itertuples():
        with Image.open(ROOT/row.png) as im:
            assert im.width==2750 and im.height==770 and abs(im.info['dpi'][0]-220)<.1
        assert '<text' in (ROOT/row.svg).read_text(encoding='utf8')
        sources=row.source_files.split(';')
        assert ';'.join(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources)==row.source_sha256
        assert row.task in {'A','B','C','D','E','F','Optional'}
    assert manifest.set_index('id').loc['F71-02','units']=='dimensionless'
    assert manifest.loc[manifest.id.isin(['F71-05']),'slide'].str.contains('notebook',case=False).all()
    tasks=pd.read_csv(ROOT/'teaching/week7_day1_task_manifest.csv')
    assert tasks.task_id.tolist()==list('ABCDEF') and tasks.minutes_including_debrief.sum()==62
    assert tasks.slide.tolist()==[row[1] for row in TASKS]
    schedule=pd.read_csv(ROOT/'teaching/week7_day1_schedule.csv')
    assert schedule.core_minutes.tolist()==SLIDE_MINUTES
    assert schedule.groupby('block').core_minutes.sum().tolist()==[50]*4
    assert pd.read_csv(OUTPUT/'tariff_cross_costs.csv',index_col=0).shape==(2,2)
    assert pd.read_csv(OUTPUT/'F71-11_native_table.csv').shape==(2,3)
    assert pd.read_csv(OUTPUT/'peak_comparison_metrics.csv').shape[0]==2
    frozen=json.loads((snapshot/'protected_hashes.json').read_text(encoding='utf8'))
    for name,digest in frozen.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    external=json.loads((snapshot/'external_hashes.json').read_text(encoding='utf8'))
    for name,digest in external.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    return len(frozen),len(external)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True,help='Private pre-edit directory containing protected_hashes.json.')
    args=parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'protected_hashes.json').write_bytes((args.snapshot/'protected_hashes.json').read_bytes())
    student,solution=static_checks()
    private=Path(tempfile.mkdtemp(prefix='BENV0169_week7_day1_validation_'))
    kernels=private/'kernels';spec=kernels/'week7-day1-validation';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 7.1 validation','language':'python'}),encoding='utf8')
    executed=execute(solution,kernels,private/'solution_checks.json')
    nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Fresh-kernel solution and independent checks passed.',flush=True)
    completed=copy.deepcopy(student)
    for a,b in zip(completed.cells,solution.cells):
        if 'task' in a.metadata.get('tags',[]):a.source=b.source
    completed=execute(completed,kernels,private/'completed_student_checks.json')
    nbformat.write(completed,private/'completed_student_PRIVATE.ipynb')
    first=json.loads((private/'solution_checks.json').read_text());second=json.loads((private/'completed_student_checks.json').read_text())
    # Dataframes can have NaNs for objective-specific columns; compare the encoded records.
    assert json.dumps(first,sort_keys=True)==json.dumps(second,sort_keys=True)
    static_checks();count,external_count=output_checks(args.snapshot)
    first.update({'fresh_kernel_runs':2,'protected_files_unchanged':count,'external_files_unchanged':external_count,
        'paired_cells':len(student.cells),'student_task_stops':6,'student_outputs':0,'figure_pairs':len(FIGURES),
        'private_evidence_directory':str(private),'pre_edit_snapshot':str(args.snapshot),
        'command':subprocess.list2cmdline([sys.executable,'-B',*sys.argv])})
    # Objective-specific dataframe columns use NaN for "not applicable".
    # Store those as JSON null, retaining every numerical result unchanged.
    def finite_json(value):
        if isinstance(value,dict):return {key:finite_json(item) for key,item in value.items()}
        if isinstance(value,list):return [finite_json(item) for item in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    (OUTPUT/'validation_results.json').write_text(json.dumps(finite_json(first),indent=2,allow_nan=False),encoding='utf8')
    print(f'PASS: completed student matches; {len(FIGURES)} figure pairs; {count} protected files unchanged.',flush=True)
    print(json.dumps({k:first[k] for k in ['solver','scheduling_value_tou_gbp','peak_lp_oracle_kw','matched_max_difference_c','failed_experiment_cases']},indent=2))


if __name__=='__main__':main()
