"""Independent checks of the notebook's shared-plan scenario formulation."""
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from course_utils.week8_stochastic import error_covariance,teaching_inputs,solve_scenario_update


def oracle(plan, model, gamma):
    """Full-state sparse HiGHS LP with one power vector and S state/slack paths."""
    weather = plan['outdoor_scenarios_c']; S, N = weather.shape
    count = N+S*N+S*(N+1)
    dt = model['sampling_interval_hours']; b = model['loss_coefficient_per_step']
    h = model['heat_coefficient_c_per_kw_step']*model['nominal_cop']
    cost = np.r_[dt*plan['price'], np.full(S*N, dt*gamma/S), np.zeros(S*(N+1))]
    eq = lil_matrix((S*(N+1), count)); rhs = np.zeros(S*(N+1))
    ub = lil_matrix((2*S*N, count)); limit = np.zeros(2*S*N)
    for r in range(S):
        t = N+S*N+r*(N+1); row = r*(N+1)
        eq[row, t] = 1.; rhs[row] = plan['temperature_scenarios_c'][r, 0]
        for j in range(N):
            eq[row+j+1, j] = -h
            eq[row+j+1, t+j] = -(1-b); eq[row+j+1, t+j+1] = 1.
            rhs[row+j+1] = b*weather[r, j]
            q = r*N+j
            ub[q, N+q] = -1.; ub[q, t+j+1] = -1.; limit[q] = -plan['lower'][j]
            ub[S*N+q, t+j+1] = 1.; limit[S*N+q] = plan['upper'][j]
    result = linprog(cost, A_ub=ub.tocsr(), b_ub=limit, A_eq=eq.tocsr(), b_eq=rhs,
                     bounds=[(0, model['heat_pump_max_electric_kw'])]*N
                            +[(0, None)]*(S*N)+[(None, None)]*(S*(N+1)), method='highs')
    assert result.success, result.message
    gap = abs(result.fun-plan['objective_gbp'])
    assert gap < 1e-6, gap
    return gap


def check_extension(model, covariance, errors, realisations, make_controller, inputs,
                    initial_plans, tiny_plans, metrics, intervals, saved_plans,
                    pairs, summary, evidence, protocol):
    """Check information timing, independent paths, objectives and Monte Carlo scores."""
    H=12; S=40; R=200; sigma=2.; rho=.8; dt=.5; gamma=25.
    assert protocol['planning_seed']==811 and protocol['evaluation_seed']==8282
    assert protocol['realisations']==R and protocol['horizon']==protocol['reporting_steps']==H
    assert protocol['innovation_sd_c']==sigma and protocol['rho']==rho
    i,j=np.indices((H,H))
    expected_covariance=sigma**2/(1-rho**2)*(rho**np.abs(i-j)-rho**(i+j+2))
    np.testing.assert_allclose(covariance,expected_covariance,atol=1e-12,rtol=0)
    expected_errors=np.random.default_rng(811).multivariate_normal(np.zeros(H),covariance,size=S)
    expected_errors-=expected_errors.mean(axis=0)
    np.testing.assert_array_equal(errors,expected_errors)
    np.testing.assert_allclose(errors.mean(axis=0),0,atol=1e-12)
    expected_paths=np.random.default_rng(8282).multivariate_normal(np.zeros(H),covariance,size=R)
    np.testing.assert_array_equal(realisations,expected_paths)
    assert not inputs['current_weather_known'] and np.std(errors[:,0])>1
    assert len(metrics)==2*R and len(intervals)==2*R*H and len(pairs)==R
    assert metrics.complete.all() and metrics.accepted_solves.eq(H).all() and metrics.fallback_count.eq(0).all()
    assert intervals.accepted.all() and not intervals.fallback.any()
    assert intervals.max_check_residual.max()<1e-6
    b=model['loss_coefficient_per_step']; h=model['heat_coefficient_c_per_kw_step']*model['nominal_cop']
    payload=[]
    for (path,label),rows in intervals.groupby(['path','strategy'],sort=False):
        assert rows.step.tolist()==list(range(H))
        start=rows.start_c.to_numpy(); end=rows.end_c.to_numpy(); power=rows.power_kw.to_numpy()
        actual=5.+realisations[path]
        previous=np.r_[0.,realisations[path,:-1]]
        np.testing.assert_allclose(rows.forecast_current_c,5.+rho*previous,atol=1e-12,rtol=0)
        np.testing.assert_array_equal(rows.actual_outdoor_c,actual)
        assert start[0]==20.
        np.testing.assert_array_equal(start[1:],end[:-1])
        np.testing.assert_allclose(end,(1-b)*start+b*actual+h*power,atol=1e-12,rtol=0)
        assert power.min()>=0 and power.max()<=5.
        bill=dt*.30*power.sum()
        below=dt*np.maximum(20.-end,0).sum(); above=dt*np.maximum(end-23.,0).sum()
        cost=bill+gamma*below
        score=metrics.loc[metrics.path.eq(path)&metrics.strategy.eq(label)].iloc[0]
        np.testing.assert_allclose([score.bill_gbp,score.penalty_gbp,score.total_cost_gbp,
            score.below_degree_hours,score.above_degree_hours],
            [bill,gamma*below,cost,below,above],atol=1e-10,rtol=0)
        payload.append({'path':int(path),'strategy':label,'bill_gbp':float(bill),
                        'lower_degree_hours':float(below),'total_cost_gbp':float(cost)})
    # Future test weather cannot affect the identical first information set.
    first=intervals.loc[intervals.step.eq(0)]
    for label,rows in first.groupby('strategy'):
        assert rows.power_kw.max()-rows.power_kw.min()<1e-5
    forecast=metrics.pivot(index='path',columns='strategy',values='total_cost_gbp')
    delta=forecast['Scenario MPC']-forecast['Mean-forecast MPC']
    np.testing.assert_allclose(pairs.delta_gbp,delta,atol=1e-12,rtol=0)
    se=delta.std(ddof=1)/np.sqrt(R)
    np.testing.assert_allclose([evidence['mean_delta_gbp'],evidence['mean_delta_ci_low'],evidence['mean_delta_ci_high']],
        [delta.mean(),delta.mean()-1.96*se,delta.mean()+1.96*se],atol=1e-12,rtol=0)
    assert evidence['fraction_lower_cost']==float((delta<0).mean())
    assert evidence['mean_delta_ci_high']<0, 'The sampled case does not support lower expected cost.'
    for row in summary.itertuples():
        group=metrics.loc[metrics.strategy.eq(row.strategy)]
        assert row.realisations==R and row.failed==row.fallbacks==0
        for metric in ['bill_gbp','penalty_gbp','total_cost_gbp','below_degree_hours','above_degree_hours']:
            np.testing.assert_allclose([getattr(row,metric+'_mean'),getattr(row,metric+'_se')],
                [group[metric].mean(),group[metric].std(ddof=1)/np.sqrt(R)],atol=1e-12,rtol=0)
    # Independent HiGHS checks on the initial, repeated-scenario and saved feedback plans.
    plans=list(initial_plans.values())+list(tiny_plans.values())+list(saved_plans.values())
    gaps=[oracle(plan,model,gamma) for plan in plans]
    assert initial_plans['Scenario MPC']['power_kw'][0]>initial_plans['Mean-forecast MPC']['power_kw'][0]
    # Zero uncertainty reduces the shared-plan problem to its deterministic counterpart.
    zero=error_covariance(2,0.,rho)
    np.testing.assert_array_equal(zero,np.zeros((2,2)))
    one_input=teaching_inputs(model,N=2)
    zero_plans=[]
    for count in [1,3]:
        c=make_controller(model,N=2,S=count,gamma=gamma)
        p=solve_scenario_update(c,one_input,np.tile(one_input['outdoor'],(count,1)))
        assert p['accepted'];zero_plans.append(p)
    np.testing.assert_allclose(zero_plans[0]['objective_gbp'],zero_plans[1]['objective_gbp'],atol=1e-6,rtol=0)
    invalid={**inputs,'upper':np.full(H,-100.)}
    c=make_controller(model,N=H,S=S,gamma=gamma)
    assert not solve_scenario_update(c,invalid,inputs['outdoor'][None,:]+errors)['accepted']
    known={**inputs,'current_weather_known':True}
    assert not solve_scenario_update(c,known,inputs['outdoor'][None,:]+errors)['accepted']
    return {'status':'PASS','scenario_count':S,'independent_weather_paths':R,'policy_runs':2*R,
        'accepted_feedback_solves':2*R*H,'independently_checked_plans':len(gaps),
        'max_objective_gap_gbp':max(gaps),'evidence':evidence,'metrics':payload,
        'checks':['independent planning/evaluation draws','Gaussian conditional covariance',
            'unobserved next-interval weather','conditional forecasts use only past weather',
            'paired common weather paths','all physical trajectories and realised objectives',
            'Monte Carlo means and standard errors over independent paths',
            'independent sparse HiGHS optima','zero-uncertainty equivalence',
            'infeasible and invalid-nowcast rejection']}
