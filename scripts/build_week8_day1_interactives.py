"""Embed verified Tutorial 8.1 runs in two offline HTML teaching figures."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linprog

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from course_utils.week8_control import (
    SimulationSession,announced_schedule,make_controller,assign_inputs,solve_update,first_action)


def independent_lp(inputs,model,gamma):
    """Check each saved optimum using a separate full-state HiGHS formulation."""
    n=len(inputs['price']);size=3*n+1
    objective=np.r_[model['sampling_interval_hours']*inputs['price'],
                    np.full(n,model['sampling_interval_hours']*gamma),np.zeros(n+1)]
    equality=np.zeros((n+1,size));rhs=np.zeros(n+1)
    equality[0,2*n]=1;rhs[0]=inputs['measurement_c']
    b=model['loss_coefficient_per_step'];h=model['heat_coefficient_c_per_kw_step']*model['nominal_cop']
    inequality=np.zeros((2*n,size));limit=np.zeros(2*n)
    for j in range(n):
        equality[j+1,j]=-h
        equality[j+1,2*n+j]=-(1-b)
        equality[j+1,2*n+j+1]=1
        rhs[j+1]=b*inputs['outdoor'][j]
        inequality[j,n+j]=-1;inequality[j,2*n+j+1]=-1;limit[j]=-inputs['lower'][j]
        inequality[n+j,2*n+j+1]=1;limit[n+j]=inputs['upper'][j]
    return linprog(objective,A_ub=inequality,b_ub=limit,A_eq=equality,b_eq=rhs,
        bounds=[(0,model['heat_pump_max_electric_kw'])]*n+[(0,None)]*n+[(None,None)]*(n+1),method='highs')


def run(mode,operation,forecasts,model):
    session=SimulationSession(operation,model,'2025-02-03',M=48,seed=88)
    schedule=announced_schedule(operation)
    controller=make_controller(model,N=48,gamma=25.)
    plans=[];max_gap=0.
    for k in range(48):
        n=48 if mode=='rolling' else 48-k
        if mode=='shrinking':controller=make_controller(model,N=n,gamma=25.)
        inputs=session.inputs(schedule,forecasts,n)
        assign_inputs(controller,inputs)
        plan=solve_update(controller,inputs)
        assert plan['accepted'],(mode,k,plan)
        check=independent_lp(inputs,model,25.)
        assert check.success,(mode,k,check.message)
        gap=abs(plan['objective_gbp']-check.fun)
        assert gap<1e-6,(mode,k,gap)
        max_gap=max(max_gap,gap)
        plans.append({'origin':k,'n':n,'measurement':inputs['measurement_c'],
            'outdoor':inputs['outdoor'].tolist(),'price':inputs['price'].tolist(),
            'lower':inputs['lower'].tolist(),'upper':inputs['upper'].tolist(),
            'power':plan['power_kw'].tolist(),'temperature':plan['temperature_c'].tolist(),
            'objective':plan['objective_gbp'],'bill':plan['bill_gbp'],'penalty':plan['penalty_gbp']})
        session.advance(first_action(plan),plan)
    result=session.result(mode)
    rows=result['intervals'];states=result['boundaries'];dt=model['sampling_interval_hours']
    metrics={'bill':float(dt*np.dot(rows.applied_power_kw,rows.price_gbp_per_kwh)),
             'energy':float(dt*rows.applied_power_kw.sum()),
             'below':float(dt*np.maximum(rows.lower_c-rows.true_end_c,0).sum()),
             'above':float(dt*np.maximum(rows.true_end_c-rows.upper_c,0).sum())}
    return {'plans':plans,'applied':rows.applied_power_kw.tolist(),
        'measured':states.measured_temperature_c.tolist(),'true':states.true_temperature_c.tolist(),
        'metrics':metrics},max_gap


def embed(path,element_id,payload):
    text=path.read_text(encoding='utf8')
    replacement=f'<script id="{element_id}" type="application/json">'+json.dumps(payload,separators=(',',':'),allow_nan=False)+'</script>'
    pattern=rf'<script id="{element_id}" type="application/json">.*?</script>'
    text,count=re.subn(pattern,lambda _:replacement,text,flags=re.S)
    assert count==1,path
    path.write_text(text,encoding='utf8')


def two_interval_oracles(model):
    defaults={'initial':20.,'forecast':5.,'nowcast':5.,'price0':.1,'price1':.4,
              'lower':20.,'upper':23.,'gamma':25.}
    settings=[defaults,{**defaults,'price0':.4,'price1':.1},
              {**defaults,'initial':22.8},{**defaults,'initial':18.,'lower':21.5},
              {**defaults,'initial':24.5},{**defaults,'gamma':0.},
              {**defaults,'upper':(1-model['loss_coefficient_per_step'])*20+model['loss_coefficient_per_step']*5}]
    rng=np.random.default_rng(8162)
    for _ in range(240):
        settings.append({'initial':float(rng.uniform(18,24.5)),'forecast':float(rng.uniform(-5,20)),
            'nowcast':float(rng.uniform(-5,20)),'price0':float(rng.uniform(.05,.6)),
            'price1':float(rng.uniform(.05,.6)),'lower':float(rng.uniform(18,22)),
            'upper':float(rng.uniform(22,25)),'gamma':float(rng.choice([0,.02,.5,2,25]))})
    cases=[]
    for setting in settings:
        inputs={'measurement_c':setting['initial'],'price':np.array([setting['price0'],setting['price1']]),
                'outdoor':np.array([setting['nowcast'],setting['forecast']]),
                'lower':np.full(2,setting['lower']),'upper':np.full(2,setting['upper'])}
        result=independent_lp(inputs,model,setting['gamma'])
        assert result.status in [0,2]
        cases.append({'input':setting,'feasible':bool(result.success),
                      'value':float(result.fun) if result.success else None})
    return cases


def main():
    data_dir=ROOT/'data/reference_building'
    model=json.loads((data_dir/'verified_controller_model.json').read_text(encoding='utf8'))
    operation=pd.read_csv(data_dir/'operation_conditions.csv',index_col='timestamp',parse_dates=True)
    forecasts=pd.read_csv(data_dir/'operation_forecasts.csv',parse_dates=['issue_time','valid_time'])
    runs={};gaps={}
    for mode in ['rolling','shrinking']:runs[mode],gaps[mode]=run(mode,operation,forecasts,model)
    np.testing.assert_allclose(runs['rolling']['applied'][0],runs['shrinking']['applied'][0],atol=1e-7)
    baseline=pd.read_csv(ROOT/'teaching/week8_day1_outputs/successive_plans_intervals.csv')
    np.testing.assert_allclose(runs['rolling']['applied'],baseline.applied_power_kw,atol=1e-6,rtol=0)
    signals=operation.loc['2025-02-03'].iloc[:48]
    following=operation.loc['2025-02-04'].iloc[:48]
    signals=pd.concat([signals,following])
    sources=['data/reference_building/'+name for name in
             ['operation_conditions.csv','operation_forecasts.csv','verified_controller_model.json']]
    sources+=['course_utils/week8_control.py','course_utils/reference_building.py']
    payload={'model':model,'dt':.5,'day':'3 February 2025','seed':88,'gamma':25.,'runs':runs,
             'signals':{'lower':signals.comfort_min_c.tolist(),'upper':signals.comfort_max_c.tolist(),
                        'price':signals.import_price_gbp_per_kwh.tolist(),
                        'outdoor':signals.outdoor_temperature_c.tolist()},
             'source_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}
    embed(ROOT/'teaching/tutorial_8_1_mpc_player.html','experiment-data',payload)
    embed(ROOT/'teaching/tutorial_8_1_two_interval_heating.html','model-data',{'model':model,'gamma':25.})
    report={'status':'PASS','saved_plans':96,'independent_solver':'SciPy HiGHS',
            'max_objective_difference_gbp':gaps,'core_trajectory_matches_notebook':True,
            'same_initial_problem':True,'rolling_horizons':[48]*48,
            'shrinking_horizons':list(range(48,0,-1)),'source_sha256':payload['source_sha256']}
    (ROOT/'teaching/week8_day1_outputs/interactives_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    (ROOT/'teaching/week8_day1_outputs/interactive_lp_oracles.json').write_text(
        json.dumps(two_interval_oracles(model),indent=2),encoding='utf8')
    print('PASS: embedded 96 independently checked plans; rolling trajectory matches Tutorial 8.1.')


if __name__=='__main__':main()
