"""Private clean-kernel checks and independently assembled LP oracles for Tutorial 7.2."""
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
from build_week7_day2 import ROOT, STEM, TASKS, FIGURES, SLIDE_MINUTES, make_notebook

OUTPUT=ROOT/'teaching/week7_day2_outputs'
CHECKS=r'''
import copy as _copy
import math as _math
import importlib.metadata as _metadata
import warnings
from scipy.optimize import linprog as _linprog
from unittest.mock import patch as _patch
from scripts.week7_day2_support import FEASIBILITY_TOL, FLOW_TOL

physical_diagnostics=case_checks.physical_diagnostics

assert N==48 and dt==.5 and len(state_time)==49
assert state_time[0]==pd.Timestamp('2025-02-03') and state_time[-1]==pd.Timestamp('2025-02-04')
assert np.all(np.diff(state_time.values)==np.timedelta64(30,'m'))
assert np.all(import_prices>export_prices) and np.all(import_prices>0)
assert problem_A.is_dcp() and problem_B.is_dcp() and problem_C.is_dcp()
_plots={r['id']:r for r in _plot_evidence}
assert len(_plots)==12
for _id,_count in {'F72-01':5,'F72-04':8,'F72-08':2,'F72-09':2,'F72-11':7,'F72-12':5}.items():
    assert _plots[_id]['power_steps']==_count,(_id,_plots[_id])
for _id,_count in {'F72-05':1,'F72-07':2,'F72-11':2}.items():
    assert _plots[_id]['boundary_traces']==_count,(_id,_plots[_id])
assert all(result['status']==cp.OPTIMAL and result['success'] for result in all_results.values())
assert all(result['peak_weight']==0 for result in core_cases.values())
assert list(core_cases)==['A','B','C']
assert np.all(case_A['pv_kw']==0)
for _r in [case_A,case_B]:
    assert not _r['battery'] and 'energy_kwh' not in _r
    np.testing.assert_array_equal(_r['charge_kw'],np.zeros(N))
    np.testing.assert_array_equal(_r['discharge_kw'],np.zeros(N))
np.testing.assert_allclose(case_A['grid_import_kw'],base_load+case_A['power_kw'],rtol=0,atol=FEASIBILITY_TOL)
np.testing.assert_allclose(case_A['grid_export_kw'],0,rtol=0,atol=FEASIBILITY_TOL)
for _r in [case_B,case_C]:np.testing.assert_array_equal(_r['pv_kw'],pv_available)
assert abs(one_kw_charge_kwh-.475)<1e-12
assert abs(one_kw_discharge_kwh-(.5/.95))<1e-12

# Independently eliminate temperature and assemble SciPy/HiGHS LPs from arrays.
# This avoids reusing CVXPY constraints or the notebook's result-check helper.
_response=np.zeros((N,N));_free=np.empty(N);_free_state=TRUE_T0
for _k in range(N):
    _free_state=(1-b)*_free_state+b*outdoor[_k];_free[_k]=_free_state
    for _j in range(_k+1):_response[_k,_j]=c*COP*(1-b)**(_k-_j)

def independent_lp(pv,battery=False,terminal=True,fixed=None,weight=0.):
    slices={};cursor=0
    for name,size in [('heat',N),('imp',N),('exp',N)]+([('ch',N),('dis',N),('energy',N+1)] if battery else [])+([('peak',1)] if weight else []):
        slices[name]=slice(cursor,cursor+size);cursor+=size
    cost=np.zeros(cursor);cost[slices['imp']]=dt*import_prices;cost[slices['exp']]=-dt*export_prices
    bounds=[(0,P_MAX)]*N+[(0,None)]*(2*N)
    if battery:bounds += [(0,BATTERY_MAX)]*(2*N)+[(E_MIN,E_MAX)]*(N+1)
    if weight:cost[slices['peak']]=weight;bounds += [(0,None)]
    equal=[];rhs=[];upper_rows=[];upper_rhs=[]
    for k in range(N):
        row=np.zeros(cursor);row[slices['imp'].start+k]=1;row[slices['exp'].start+k]=-1;row[slices['heat'].start+k]=-1
        if battery:row[slices['dis'].start+k]=1;row[slices['ch'].start+k]=-1
        equal.append(row);rhs.append(base_load[k]-pv[k])
        high=np.zeros(cursor);high[slices['heat']]=_response[k]
        upper_rows += [high,-high];upper_rhs += [upper[k]-_free[k],_free[k]-lower[k]]
        if battery:
            row=np.zeros(cursor);row[slices['energy'].start+k+1]=1;row[slices['energy'].start+k]=-1
            row[slices['ch'].start+k]=-dt*ETA_CHARGE;row[slices['dis'].start+k]=dt/ETA_DISCHARGE
            equal.append(row);rhs.append(0.)
        if fixed is not None:
            row=np.zeros(cursor);row[slices['heat'].start+k]=1;equal.append(row);rhs.append(fixed[k])
        if weight:
            row=np.zeros(cursor);row[slices['imp'].start+k]=1;row[slices['peak'].start]=-1
            upper_rows.append(row);upper_rhs.append(0.)
    if battery:
        row=np.zeros(cursor);row[slices['energy'].start]=1;equal.append(row);rhs.append(E_INIT)
        if terminal:
            row=np.zeros(cursor);row[slices['energy'].stop-1]=1;row[slices['energy'].start]=-1
            equal.append(row);rhs.append(0.)
    lp=_linprog(cost,A_ub=upper_rows,b_ub=upper_rhs,A_eq=equal,b_eq=rhs,bounds=bounds,method='highs')
    assert lp.success,lp.message
    return float(lp.fun)

_oracle_values={}
for _name,_r in all_results.items():
    _oracle_values[_name]=independent_lp(_r['pv_kw'],_r['battery'],bool(_r['terminal']),
        case_B['power_kw'] if _name=='fixed_heating' else None,_r['peak_weight'])
    np.testing.assert_allclose(_r['solver_objective'],_oracle_values[_name],rtol=0,atol=FEASIBILITY_TOL)
_heating_lp=_linprog(dt*import_prices,A_ub=np.vstack([_response,-_response]),
    b_ub=np.r_[upper-_free,_free-lower],bounds=[(0,P_MAX)]*N,method='highs')
assert _heating_lp.success
_heating_bill=dt*_math.fsum(float(p)*float(w) for p,w in zip(import_prices,case_A['power_kw']))
np.testing.assert_allclose(_heating_bill,_heating_lp.fun,rtol=0,atol=FEASIBILITY_TOL)
np.testing.assert_allclose(case_A['bill_gbp']-dt*np.dot(import_prices,base_load),_heating_lp.fun,rtol=0,atol=FEASIBILITY_TOL)
# A live Day 1 output, if available, is corroboration only; execution never imports Day 1.
_day1=ROOT/'teaching/week7_day1_outputs/hard_summary.csv'
_day1_bill=None
if _day1.is_file():
    _day1_bill=float(pd.read_csv(_day1).bill_gbp.iloc[0])
    np.testing.assert_allclose(_heating_bill,_day1_bill,rtol=0,atol=FEASIBILITY_TOL)

# Physical and performance checks use scalar recurrences and math.fsum independently.
_metrics=[]
for _name,_r in all_results.items():
    _p=_r['power_kw'];_t=_r['temperature_c'];_imp=_r['grid_import_kw'];_exp=_r['grid_export_kw']
    _ch=_r['charge_kw'];_dis=_r['discharge_kw']
    for _key in ['power_kw','grid_import_kw','grid_export_kw','charge_kw','discharge_kw','pv_kw']:
        assert _r[_key].shape==(N,) and np.isfinite(_r[_key]).all()
    assert _t.shape==(N+1,) and np.isfinite(_t).all()
    _replay=[TRUE_T0]
    for _k in range(N):_replay.append(_replay[-1]+b*(outdoor[_k]-_replay[-1])+c*COP*_p[_k])
    np.testing.assert_allclose(_t,_replay,rtol=0,atol=FEASIBILITY_TOL)
    assert min(_p)>=-FEASIBILITY_TOL and max(_p)<=P_MAX+FEASIBILITY_TOL
    assert np.all(_t[1:]>=lower-FEASIBILITY_TOL) and np.all(_t[1:]<=upper+FEASIBILITY_TOL)
    for _k in range(N):
        assert abs(_imp[_k]+_r['pv_kw'][_k]+_dis[_k]-base_load[_k]-_p[_k]-_ch[_k]-_exp[_k])<=FEASIBILITY_TOL
    assert min(_imp)>=-FEASIBILITY_TOL and min(_exp)>=-FEASIBILITY_TOL
    assert not any(i>FLOW_TOL and e>FLOW_TOL for i,e in zip(_imp,_exp))
    assert not any(c0>FLOW_TOL and d0>FLOW_TOL for c0,d0 in zip(_ch,_dis))
    _bill=dt*_math.fsum(float(pi)*float(i)-float(pe)*float(e) for pi,i,pe,e in zip(import_prices,_imp,export_prices,_exp))
    _row=performance_metrics(_r)
    _independent={'bill_gbp':_bill,'import_kwh':dt*_math.fsum(map(float,_imp)),
        'export_kwh':dt*_math.fsum(map(float,_exp)),'heat_pump_kwh':dt*_math.fsum(map(float,_p)),
        'peak_import_kw':max(_imp),'mean_endpoint_temperature_c':_math.fsum(map(float,_t[1:]))/N,
        'final_temperature_c':_t[-1],
        'below_band_degree_hours':dt*_math.fsum(max(float(l)-float(t),0.) for l,t in zip(lower,_t[1:])),
        'above_band_degree_hours':dt*_math.fsum(max(float(t)-float(u),0.) for t,u in zip(_t[1:],upper))}
    if _r['battery']:
        _e=_r['energy_kwh'];assert _e.shape==(N+1,) and np.isfinite(_e).all()
        _energy_replay=[E_INIT]
        for _cc,_dd in zip(_ch,_dis):_energy_replay.append(_energy_replay[-1]+dt*ETA_CHARGE*_cc-dt*_dd/ETA_DISCHARGE)
        np.testing.assert_allclose(_e,_energy_replay,rtol=0,atol=FEASIBILITY_TOL)
        assert min(_e)>=E_MIN-FEASIBILITY_TOL and max(_e)<=E_MAX+FEASIBILITY_TOL
        assert min(_ch)>=-FEASIBILITY_TOL and min(_dis)>=-FEASIBILITY_TOL
        assert max(_ch)<=BATTERY_MAX+FEASIBILITY_TOL and max(_dis)<=BATTERY_MAX+FEASIBILITY_TOL
        if _r['terminal']:assert abs(_e[-1]-_e[0])<=FEASIBILITY_TOL
        _charged=dt*_math.fsum(map(float,_ch));_delivered=dt*_math.fsum(map(float,_dis))
        _loss=(1-ETA_CHARGE)*_charged+(1/ETA_DISCHARGE-1)*_delivered
        assert abs(_charged-_delivered-(_e[-1]-_e[0])-_loss)<=FEASIBILITY_TOL
        _independent.update(initial_energy_kwh=_e[0],final_energy_kwh=_e[-1],battery_throughput_kwh=_charged+_delivered)
    else:
        assert all(pd.isna(_row[k]) for k in ['initial_energy_kwh','final_energy_kwh','battery_throughput_kwh'])
    for _key,_value in _independent.items():np.testing.assert_allclose(_row[_key],_value,rtol=0,atol=1e-10)
    np.testing.assert_allclose(_r['bill_gbp'],_bill,rtol=0,atol=1e-12)
    _metrics.append(_row)
np.testing.assert_allclose(import_cost,dt*_math.fsum(float(p)*float(i) for p,i in zip(import_prices,case_C['grid_import_kw'])),rtol=0,atol=1e-12)
np.testing.assert_allclose(export_revenue,dt*_math.fsum(float(p)*float(e) for p,e in zip(export_prices,case_C['grid_export_kw'])),rtol=0,atol=1e-12)
np.testing.assert_allclose(net_bill_C,case_C['bill_gbp'],rtol=0,atol=1e-12)
np.testing.assert_allclose(charging_energy,dt*_math.fsum(map(float,case_C['charge_kw'])),rtol=0,atol=1e-12)
np.testing.assert_allclose(discharged_energy,dt*_math.fsum(map(float,case_C['discharge_kw'])),rtol=0,atol=1e-12)
np.testing.assert_allclose(stored_energy_change,case_C['energy_kwh'][-1]-case_C['energy_kwh'][0],rtol=0,atol=1e-12)
np.testing.assert_allclose(battery_losses,(1-ETA_CHARGE)*charging_energy+(1/ETA_DISCHARGE-1)*discharged_energy,rtol=0,atol=FEASIBILITY_TOL)
np.testing.assert_allclose(operation_summary.Value.to_numpy(),[import_cost,export_revenue,net_bill_C,charging_energy,discharged_energy,battery_losses,case_C['energy_kwh'][0],case_C['energy_kwh'][-1]],rtol=0,atol=1e-12)
for _key,_value in baseline_metrics.items():
    np.testing.assert_allclose(_value,performance_metrics(case_A)[_key],rtol=0,atol=1e-12)
pd.testing.assert_frame_equal(asset_comparison,pd.DataFrame([performance_metrics(r) for r in core_cases.values()]))

# Feasible-set inclusion: embed B with idle storage, keeping its heating/grid arrays.
_idle=_copy.deepcopy(case_B);_idle.update(battery=True,terminal=True,energy_kwh=np.full(N+1,E_INIT))
assert physical_diagnostics(_idle)['pass'].all()
np.testing.assert_allclose(np.diff(_idle['energy_kwh']),0,atol=0)
assert case_C['bill_gbp']<=case_B['bill_gbp']+FEASIBILITY_TOL
assert case_C['bill_gbp']<=fixed_heating['bill_gbp']+FEASIBILITY_TOL
assert no_terminal['bill_gbp']<=case_C['bill_gbp']+FEASIBILITY_TOL
np.testing.assert_allclose(fixed_heating['power_kw'],case_B['power_kw'],rtol=0,atol=FEASIBILITY_TOL)
_closed,_closed_vars=build_site(pv_available,battery=True)
_open,_open_vars=build_site(pv_available,battery=True,terminal=False)
assert len(_closed.constraints)==len(_open.constraints)+1
_no_bat,_no_bat_vars=build_site(pv_available)
assert 'energy_kwh' not in _no_bat_vars and not hasattr(_no_bat_vars['charge_kw'],'value')
np.testing.assert_array_equal(_no_bat_vars['charge_kw'],np.zeros(N))
np.testing.assert_array_equal(_no_bat_vars['discharge_kw'],np.zeros(N))

# Copied results survive re-solves and do not share writable memory.
_saved={name:{k:v.copy() for k,v in r.items() if isinstance(v,np.ndarray)} for name,r in all_results.items()}
_repeat=require_checked(optimise_site(np.zeros(N),case='Private repeated A'))
for _name,_arrays in _saved.items():
    for _key,_value in _arrays.items():
        np.testing.assert_array_equal(all_results[_name][_key],_value)
        if _key in _repeat:assert not np.shares_memory(_repeat[_key],all_results[_name][_key])
assert not np.shares_memory(case_C['power_kw'],P.value)
assert not np.shares_memory(case_C['energy_kwh'],E.value)

# Failure records cannot expose stale arrays, including exceptions on a previously solved problem.
_v=cp.Variable();_bad=cp.Problem(cp.Minimize(_v),[_v>=1,_v<=0])
_failed=solve_checked(_bad,{},pv=np.zeros(N),case='Infeasible fixture')
assert not _failed['success'] and _failed['status']==cp.INFEASIBLE and 'power_kw' not in _failed
with _patch.object(problem_C,'solve',side_effect=RuntimeError('forced failure')):
    _exception=solve_checked(problem_C,{'power_kw':P},pv=pv_available,battery=True,case='Exception fixture')
assert _exception['status']=='solver_exception' and 'power_kw' not in _exception
_nonconvex=cp.Problem(cp.Minimize(cp.multiply(_v,_v)),[])
assert solve_checked(_nonconvex,{},pv=np.zeros(N))['status']=='not_dcp'
class _Candidate:
    status=cp.OPTIMAL_INACCURATE
    value=case_C['bill_gbp']
    def is_dcp(self):return True
    def solve(self,**kwargs):pass
_candidate_arrays={key:value.copy() for key,value in _saved['C'].items() if key!='pv_kw'}
with warnings.catch_warnings(record=True) as _warnings:
    _inaccurate=solve_checked(_Candidate(),_candidate_arrays,pv=pv_available,battery=True,case='Private inaccurate fixture')
assert _inaccurate['success'] and _inaccurate['status']==cp.OPTIMAL_INACCURATE and _warnings
_candidate_arrays['energy_kwh'][-1]-=1
with warnings.catch_warnings(record=True):
    _invalid=solve_checked(_Candidate(),_candidate_arrays,pv=pv_available,battery=True)
assert not _invalid['success'] and not all(row['pass'] for row in _invalid['diagnostics'])
_candidate_arrays['power_kw'][0]=np.nan
assert not solve_checked(_Candidate(),_candidate_arrays,pv=pv_available,battery=True)['success']
_candidate_arrays['power_kw']=np.zeros(N-1)
assert not solve_checked(_Candidate(),_candidate_arrays,pv=pv_available,battery=True)['success']
_overlap=_copy.deepcopy(case_C);_overlap['charge_kw'][0]=1.;_overlap['discharge_kw'][0]=1.
assert physical_diagnostics(_overlap).set_index('check').loc['Charge/discharge overlap','value']>=1
_overlap=_copy.deepcopy(case_C);_overlap['grid_import_kw'][0]+=1;_overlap['grid_export_kw'][0]+=1
assert physical_diagnostics(_overlap).set_index('check').loc['Import/export overlap','value']>=1

# Optional event uses the same assets and exactly 6 half-hours, without forcing a sign.
assert list(event_table.Intervals)==[32,6,10]
assert sum(int(mask.sum()) for mask in masks.values())==N
np.testing.assert_allclose(event_table['Difference (kWh)'],event_table['Cost-only (kWh)']-event_table['Peak-weighted (kWh)'],rtol=0,atol=1e-12)
np.testing.assert_allclose(event_table['Difference (kWh)'].sum(),dt*(case_C['grid_import_kw']-peak_weighted['grid_import_kw']).sum(),rtol=0,atol=1e-12)

validation_payload={'status':'PASS','solver':SOLVER,'packages':{p:_metadata.version(p) for p in ['cvxpy','clarabel','numpy','pandas','scipy','matplotlib','nbclient']},
    'independent_lp_objectives_gbp':_oracle_values,'heating_only_oracle_gbp':float(_heating_lp.fun),
    'day1_corroborating_heating_bill_gbp':_day1_bill,'metrics':_metrics,
    'terminal_diagnostic':terminal_table.to_dict(orient='records'),'event_metrics':event_table.to_dict(orient='records'),
    'figure_geometry':_plot_evidence,
    'no_terminal_constraint_difference':len(_closed.constraints)-len(_open.constraints),
    'failure_fixture_status':_failed['status'],'exception_fixture_status':_exception['status'],
    'checks':'PASS: independent six-case HiGHS oracles; thermal/site/battery recurrences; losses; metrics; feasibility inclusion; shapes; copied arrays; failures; inaccurate statuses; simultaneous flows; six-interval event'}
'''


def static_checks():
    student=nbformat.read(ROOT/(STEM+'.ipynb'),4)
    solution=nbformat.read(ROOT/(STEM+'_solution.ipynb'),4)
    assert len(student.cells)==len(solution.cells)
    assert [c.id for c in student.cells]==[c.id for c in solution.cells]
    for notebook,is_solution in [(student,False),(solution,True)]:
        canonical=make_notebook(is_solution)
        assert [(c.id,c.source,c.cell_type) for c in notebook.cells]==[(c.id,c.source,c.cell_type) for c in canonical.cells]
        for cell in notebook.cells:
            assert not any(word in cell.source for word in ['FEASIBILITY_TOL','FLOW_TOL','physical_diagnostics','site_residual','battery_residual','optimal_inaccurate','is_dcp'])
            if cell.cell_type=='code':
                compile(cell.source,cell.id,'exec')
                if not is_solution:assert cell.execution_count is None and not cell.outputs
                assert 'course_utils.control' not in cell.source
    task_cells=[c for c in student.cells if 'task' in c.metadata.get('tags',[])]
    assert [c.metadata.tags[-1] for c in task_cells]==list('ABCDEF')
    for cell in task_cells:
        try:exec(cell.source,{})
        except NotImplementedError as error:assert 'Task '+cell.metadata.tags[-1] in str(error)
        else:raise AssertionError('Student scaffold must stop: '+cell.id)
    for a,b in zip(student.cells,solution.cells):
        if 'task' not in a.metadata.get('tags',[]) and not a.id.startswith('w72-response-'):
            assert a.source==b.source
    assert sum(SLIDE_MINUTES)==200
    assert [sum(SLIDE_MINUTES[i:i+8]) for i in range(0,32,8)]==[50]*4
    assert sum(t[2] for t in TASKS)==70
    return student,solution


def execute(notebook,kernels,result_path):
    work=copy.deepcopy(notebook)
    check_source=CHECKS+'\nPath('+repr(str(result_path))+').write_text(json.dumps(validation_payload,indent=2),encoding="utf8")'
    check_source+='\nfor _case_key,_case_result in all_results.items():\n    physical_diagnostics(_case_result).to_csv(Path('+repr(str(result_path.parent))+')/(_case_key+"_physical_checks_PRIVATE.csv"),index=False)'
    work.cells.append(nbformat.v4.new_code_cell(check_source,id='private-validation'))
    # Validate every export's geometry while its Matplotlib artists still exist.
    probe=r'''
_original_save=save_figure
_plot_evidence=[]
def save_figure(fig,figure_id,*args,**kwargs):
    from matplotlib.patches import StepPatch
    from matplotlib.dates import date2num
    assert tuple(fig.get_size_inches())==((12.5,7.) if figure_id=='F72-04' else (12.5,3.5))
    steps=states=0
    for ax in fig.axes:
        for patch in ax.patches:
            if isinstance(patch,StepPatch):
                edges=patch.get_data().edges
                assert len(edges)==49
                np.testing.assert_array_equal(edges,date2num(state_time))
                steps+=1
        for line in ax.lines:
            x=np.asarray(line.get_xdata())
            if len(x)==49:
                np.testing.assert_array_equal(pd.DatetimeIndex(x),state_time);states+=1
    _plot_evidence.append({'id':figure_id,'power_steps':steps,'boundary_traces':states})
    _original_save(fig,figure_id,*args,**kwargs)
'''
    position=next(i for i,c in enumerate(work.cells) if c.id=='w72-plot-helpers')+1
    work.cells.insert(position,nbformat.v4.new_code_cell(probe,id='private-plot-checks'))
    manager=KernelManager(kernel_name='week7-day2-validation',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]))
    NotebookClient(work,km=manager,timeout=240,resources={'metadata':{'path':str(ROOT)}}).execute()
    work.cells=[c for c in work.cells if not c.id.startswith('private-')]
    return work


def output_checks(snapshot):
    from PIL import Image
    manifest=pd.read_csv(ROOT/'teaching/week7_day2_figure_manifest.csv')
    assert set(manifest.id)=={row[0] for row in FIGURES} and len(manifest)==len(FIGURES)
    for row in FIGURES:
        record=manifest.set_index('id').loc[row[0]]
        assert record.slide==row[1] and record.task==row[4]
        for field in ['png','svg']:assert (ROOT/record[field]).is_file()
        with Image.open(ROOT/record.png) as im:assert im.size==((2750,1540) if row[0]=='F72-04' else (2750,770))
        assert '<text' in (ROOT/record.svg).read_text(encoding='utf8')
    schedule=pd.read_csv(ROOT/'teaching/week7_day2_schedule.csv').fillna('')
    assert schedule.slide.tolist()==list(range(1,37))
    assert schedule.core_minutes.tolist()==SLIDE_MINUTES+[0]*4
    assert schedule.loc[schedule.block>0].groupby('block').core_minutes.sum().tolist()==[50]*4
    assert schedule.loc[schedule.slide==16,'break_after_minutes'].iloc[0]=='15–20 additional'
    task_manifest=pd.read_csv(ROOT/'teaching/week7_day2_task_manifest.csv')
    assert task_manifest[['task_id','slide','minutes_including_debrief']].values.tolist()==[list(t) for t in TASKS]
    assert pd.read_csv(OUTPUT/'asset_comparison_metrics.csv').shape==(3,13)
    assert pd.read_csv(OUTPUT/'F72-10_native_table.csv').shape==(3,7)
    assert pd.read_csv(OUTPUT/'C_operating_summary.csv').shape==(8,3)
    for name in ['A','B','C','no_terminal','fixed_heating','peak_weighted']:
        assert len(pd.read_csv(OUTPUT/(name+'_intervals.csv')))==48
        assert len(pd.read_csv(OUTPUT/(name+'_states.csv')))==49
    protected=json.loads((snapshot/'protected_hashes.json').read_text(encoding='utf8'))
    for name,digest in protected.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    return len(protected)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    student,solution=static_checks()
    private=ROOT/'tmp'/('week7_day2_validation_'+uuid.uuid4().hex)
    spec=private/'kernels/week7-day2-validation';spec.mkdir(parents=True)
    (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Week 7.2 validation','language':'python'}),encoding='utf8')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    executed=execute(solution,spec.parent,private/'solution_checks.json')
    nbformat.write(executed,ROOT/(STEM+'_solution.ipynb'))
    print('Fresh-kernel solution passed independent numerical and figure checks.',flush=True)
    completed=copy.deepcopy(student)
    for a,b in zip(completed.cells,solution.cells):
        if 'task' in a.metadata.get('tags',[]):a.source=b.source
    executed_student=execute(completed,spec.parent,private/'student_checks.json')
    nbformat.write(executed_student,private/'completed_student_PRIVATE.ipynb')
    first=json.loads((private/'solution_checks.json').read_text(encoding='utf8'))
    second=json.loads((private/'student_checks.json').read_text(encoding='utf8'))
    assert json.dumps(first,sort_keys=True)==json.dumps(second,sort_keys=True)
    static_checks();protected=output_checks(args.snapshot)
    first.update(fresh_kernel_runs=2,paired_cells=len(student.cells),student_outputs=0,student_tasks=6,
        protected_files_unchanged=protected,figure_pairs=len(FIGURES),private_evidence_directory=str(private),
        pre_edit_snapshot=str(args.snapshot),teaching_minutes=sum(SLIDE_MINUTES),task_minutes=sum(t[2] for t in TASKS),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ['scripts/build_week7_day2.py','scripts/validate_week7_day2.py','scripts/week7_day2_support.py',STEM+'.ipynb',STEM+'_solution.ipynb']})
    def finite(value):
        if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
        if isinstance(value,list):return [finite(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        return value
    (OUTPUT/'validation_results.json').write_text(json.dumps(finite(first),indent=2,allow_nan=False),encoding='utf8')
    print(f'PASS: 2 fresh kernels; {len(student.cells)} paired cells; 6 tasks; {len(FIGURES)} PNG/SVG pairs; {protected} protected files unchanged.',flush=True)
    print(json.dumps(first['independent_lp_objectives_gbp'],indent=2))


if __name__=='__main__':main()
