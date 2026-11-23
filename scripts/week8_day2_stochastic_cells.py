"""Paired guided cells for an introductory scenario-MPC extension."""
import textwrap
import nbformat as nbf

EXTENSION_TASKS = [('G', 8, 2), ('H', 17, 4)]
EXTENSION_FIGURES = [('F82-14', 'scenario_plans'), ('F82-15', 'scenario_performance'),
                     ('F82-16', 'monte_carlo_means')]


def extension_cells(solution=False):
    cells = []
    def md(key, text):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip(), id='w82-'+key))
    def code(key, text, tags=None):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip(), id='w82-'+key,
                                        metadata={'tags': tags or ['supplied']}))
    def task(letter, answer, starter):
        code('task-'+letter.lower(), answer if solution else starter, ['task', 'extension', letter])


    md('stochastic-intro', r"""
    ## Extension: stochastic MPC and expected operating cost (40 min)

    Represent uncertain weather by S=40 equally weighted scenarios and choose
    **one heating plan shared across those scenarios**. Apply its first action,
    observe the next temperature and replan. Evaluate both controllers over 200
    independent weather realisations to estimate their expected operating costs.

    Reuse **Week 4.2, U7 and Extension E2**: generate complete error vectors
    from a multivariate Gaussian model and add them to a point forecast. We use
    a controlled winter example with known error parameters, in degrees Celsius.
    The synthetic case makes the role of uncertainty explicit.

    G takes 8 min and H takes 17 min, including debriefs. Allow 15 min for the
    supplied checks, Monte Carlo evaluation and interpretation. This 40-minute
    extension follows the 192-minute core.
    """)
    md('stochastic-case', r"""
    ### A forecast before the next interval's weather is realised

    The core gave the controller an exact current outdoor input. Here heating
    is chosen **before the upcoming interval's average weather is known**.
    At boundary k we know the indoor temperature and the previous interval's
    weather. During interval k, the outdoor-temperature error evolves as

    $$T_{o,k}=5+e_k,\qquad
    e_k=\rho e_{k-1}+\underbrace{\color{red}{\sigma z_{k+1}}}_{\text{new weather innovation}},
    \quad z_{k+1}\sim\mathcal N(0,1),\quad e_{-1}=0.$$

    [k labels the operating interval; the innovation and interval-average weather
    become known at boundary k+1, after the heating action.] The next indoor
    reading is exact. The physical model and actuator remain unchanged.

    The conditional mean forecast at origin k is

    $$\widehat T_{o,k+j|k}=5+\rho^{j+1}e_{k-1},\qquad j=0,\ldots,H-1.$$

    Use rho=0.8 and innovation standard deviation sigma=2 °C. These are
    illustrative stress-case parameters, independent of the historical weather
    CSV. The six-hour planning and reporting horizons have H=M=12 half-hour
    intervals, constant price 0.30 £/kWh and comfort bounds 20–23 °C. Keep the
    5 kW limit, COP, fitted coefficients, initial 20 °C and gamma=25.

    The covariance matrix of the conditional forecast errors is supplied.
    As in Week 4's multivariate example, its off-diagonal entries describe
    dependence across future intervals. The first interval is uncertain too.
    """)
    code('stochastic-setup', r"""
    import cvxpy as cp
    from course_utils.week8_control import make_controller
    from course_utils.week8_stochastic import (
        error_covariance,teaching_inputs,solve_scenario_update,monte_carlo_mpc)
    HORIZON=REPORTING_STEPS=12
    S=40
    N_REALISATIONS=200
    SIGMA_WEATHER=2.
    RHO=.8
    PLANNING_SEED=811
    EVALUATION_SEED=8282
    weather_covariance=error_covariance(HORIZON,SIGMA_WEATHER,RHO)
    stochastic_inputs=teaching_inputs(model,N=HORIZON,rho=RHO)
    initial_origin=stochastic_inputs['origin']
    stochastic_protocol={'planning_scenarios':S,'realisations':N_REALISATIONS,
        'planning_seed':PLANNING_SEED,'evaluation_seed':EVALUATION_SEED,
        'innovation_sd_c':SIGMA_WEATHER,'rho':RHO,'horizon':HORIZON,
        'reporting_steps':REPORTING_STEPS,'gamma':gamma,'price_gbp_per_kwh':.30,
        'primary_score':'bill + gamma * lower-comfort degree-hours',
        'weather_timing':'upcoming interval weather revealed after the action',
        'case_status':'controlled illustration; parameters checked on separate development paths'}
    export_json(stochastic_protocol,'stochastic_evaluation_protocol.json')
    print('Fixed protocol:',S,'planning scenarios;',N_REALISATIONS,'independent evaluation paths')
    """)
    md('stochastic-errors', r"""
    ### G — sample complete joint weather-error vectors (8 min)

    Use the supplied covariance and `rng.multivariate_normal`, as in Week 4.2
    Extension E2, to draw S complete error vectors of length H. Then add each
    row to the current mean forecast:

    $$\boldsymbol\epsilon^{(r)}\sim\mathcal N(\boldsymbol0,\Sigma),\qquad
    \widehat T^{(r)}_{o,k+j|k}=\widehat T_{o,k+j|k}
       +\underbrace{\color{red}{\epsilon^{(r)}_j}}_{\text{scenario forecast error}},
    \quad j=0,\ldots,H-1.$$

    [k+j is the target interval, k is the issue time and r labels a complete
    possible weather path.] The scenario array has shape `(S,H)`.

    Subtract the sampled mean from each column of `sampled_errors`. This makes
    the finite scenario set's mean equal the known conditional mean forecast.
    Both policies then receive the same mean, so the comparison isolates how
    they account for the weather spread. Signed temperatures remain valid.
    Use 6 min to code and 2 min to explain the complete-vector construction.
    """)
    task('G', r"""
    rng=np.random.default_rng(PLANNING_SEED)
    sampled_errors=rng.multivariate_normal(np.zeros(HORIZON),weather_covariance,size=S)
    sampled_errors-=sampled_errors.mean(axis=0)
    outdoor_scenarios=stochastic_inputs['outdoor'][None,:]+sampled_errors
    assert outdoor_scenarios.shape==(S,HORIZON)
    np.testing.assert_allclose(outdoor_scenarios.mean(axis=0),stochastic_inputs['outdoor'],atol=1e-12)
    print('Scenario matrix:',outdoor_scenarios.shape)
    """, r"""
    rng=np.random.default_rng(PLANNING_SEED)
    # TODO G: draw S complete vectors with rng.multivariate_normal.
    sampled_errors=None
    # TODO: centre each column, then add the complete vectors to the point forecast.
    outdoor_scenarios=None
    raise NotImplementedError('G: construct the (S,HORIZON) joint weather scenarios.')
    """)
    md('stochastic-formulation', r'''
    ### H — optimise one heating plan across scenarios (17 min)

    Use a common power vector $P_{\tau|k}$ and a temperature/slack vector for
    each scenario. With probability 1/S per scenario, minimise the bill plus
    the **average predicted lower-comfort penalty**:

    $$\min_{P,\widehat T^{(r)},a^{(r)}}\Delta t\sum_{\tau=k}^{k+N-1}
      \left[p_\tau P_{\tau|k}+\frac{\gamma}{S}\sum_{r=1}^S a^{(r)}_{\tau|k}\right].$$

    For every scenario and planned interval, impose

    $$\widehat T^{(r)}_{k|k}=T_k^{\mathrm{meas}},\qquad
    \widehat T^{(r)}_{\tau+1|k}=(1-b)\widehat T^{(r)}_{\tau|k}
       +b\widehat T^{(r)}_{o,\tau|k}+c\,\mathrm{COP}\,P_{\tau|k},$$

    $$0\le P_{\tau|k}\le5,\quad a^{(r)}_{\tau|k}\ge0,\quad
    \widehat T^{(r)}_{\tau+1|k}+a^{(r)}_{\tau|k}\ge L^{\mathrm{end}}_\tau,
    \quad \widehat T^{(r)}_{\tau+1|k}\le U^{\mathrm{end}}_\tau.$$

    [Code `P[j]` is $P_{k+j|k}$; `T[r,j]` predicts boundary k+j;
    `slack[r,j]` is the lower-comfort slack for interval k+j.]
    Retain the fitted model, use N=HORIZON=12 and γ=25, and keep the existing comfort convention.

    | Code | Shape | Meaning |
    |---|---|---|
    | `P` | `(N,)` | One common electrical heating plan |
    | `T` | `(S,N+1)` | Predicted indoor-temperature paths |
    | `slack` | `(S,N)` | Scenario-specific lower shortfalls |
    | `outdoor` | `(S,N)` | Weather scenarios supplied as a parameter |

    Complete the variables, scenario constraints and objective. Divide the
    **slack sum** by S; the common electricity bill appears once. Use 13 min to
    code and 4 min to explain why separate heating plans for fully known weather
    paths would assume additional future information.

    **Hand check:** if one interval has lower shortfalls 0 and 0.4 °C in two
    equally likely scenarios, its expected penalty is
    $25\times0.5\times(0+0.4)/2=2.50$ £. These are predicted shortfalls across
    possible outcomes; the building experiences one realised temperature.
    ''')
    answer = textwrap.dedent(r'''
    def make_scenario_controller(model,N=12,S=40,gamma=25.):
        dt=model['sampling_interval_hours']
        b=model['loss_coefficient_per_step']
        h=model['heat_coefficient_c_per_kw_step']*model['nominal_cop']
        pmax=model['heat_pump_max_electric_kw']
        P=cp.Variable(N)
        T=cp.Variable((S,N+1))
        slack=cp.Variable((S,N),nonneg=True)
        initial=cp.Parameter()
        outdoor=cp.Parameter((S,N))
        price=cp.Parameter(N,nonneg=True)
        lower=cp.Parameter(N);upper=cp.Parameter(N)
        constraints=[P>=0,P<=pmax]
        for r in range(S):
            constraints += [T[r,0]==initial,
                T[r,1:]==(1-b)*T[r,:-1]+b*outdoor[r,:]+h*P,
                T[r,1:]+slack[r,:]>=lower,T[r,1:]<=upper]
        bill=dt*cp.sum(cp.multiply(price,P))
        expected_penalty=dt*gamma*cp.sum(slack)/S
        problem=cp.Problem(cp.Minimize(bill+expected_penalty),constraints)
        return {'problem':problem,'P':P,'T':T,'s':slack,'initial':initial,
            'outdoor':outdoor,'price':price,'lower':lower,'upper':upper,
            'N':N,'S':S,'model':dict(model),'gamma':gamma}
    ''').strip()
    starter = answer.replace('P=cp.Variable(N)', '# TODO H: create the common power vector.\n    P=None')
    starter = starter.replace('T=cp.Variable((S,N+1))', '# TODO: create the scenario state and slack arrays.\n    T=None')
    starter = starter.replace('slack=cp.Variable((S,N),nonneg=True)', 'slack=None')
    a=starter.index('    constraints='); b=starter.index('    problem=')
    starter=starter[:a]+'''    # TODO: add common actuator bounds and a loop over scenario constraints.
    constraints=[]
    # TODO: electricity bill once + average lower-comfort penalty.
    bill=None
    expected_penalty=None
    raise NotImplementedError('H: formulate the shared-plan scenario LP.')
'''+starter[b:]
    task('H', answer, starter)
    md('stochastic-check-text', r'''
    ### Check the formulation, then compare one plan

    A single scenario gives the deterministic problem for that weather path.
    Repeating an identical scenario should preserve the optimum: averaging
    repeated penalties leaves the objective unchanged. The supplied two-step
    example requires slack and tests both properties, including the factor 1/S.

    Then compare the 40-scenario initial problem with a one-path problem using
    their **mean weather**. Both receive the same mean forecast correction;
    their difference comes from how they account for the weather spread.
    The upper bound is imposed on every sampled path. Lower shortfalls enter
    the average penalty. Finite sampled constraints give an empirical
    approximation; probability guarantees need further assumptions and analysis.
    ''')
    code('stochastic-checks', r'''
    from course_utils.week8_control import assign_inputs,solve_update
    tiny_inputs={'origin':initial_origin,'measurement_c':16.,
        'outdoor':np.array([0.,0.]),'price':np.array([.18,.29]),
        'lower':np.array([20.,20.]),'upper':np.array([23.,23.]),
        'valid_time':pd.date_range(initial_origin,periods=2,freq='30min')}
    deterministic_tiny=make_controller(model,N=2,gamma=gamma)
    assign_inputs(deterministic_tiny,tiny_inputs)
    deterministic_tiny_plan=solve_update(deterministic_tiny,tiny_inputs)
    tiny_plans={}
    for count in [1,3]:
        c=make_scenario_controller(model,N=2,S=count,gamma=gamma)
        p=solve_scenario_update(c,tiny_inputs,np.tile(tiny_inputs['outdoor'],(count,1)))
        assert p['accepted'] and deterministic_tiny_plan['accepted']
        np.testing.assert_allclose(p['objective_gbp'],deterministic_tiny_plan['objective_gbp'],atol=1e-6,rtol=0)
        tiny_plans[count]=p
    assert deterministic_tiny_plan['penalty_gbp']>0
    assert np.isclose(dt*gamma*np.mean([0.,.4]),2.5)
    mean_weather=outdoor_scenarios.mean(axis=0,keepdims=True)
    stochastic_initial_plans={}
    for label,weather in [('Mean-forecast MPC',mean_weather),('Scenario MPC',outdoor_scenarios)]:
        c=make_scenario_controller(model,N=HORIZON,S=len(weather),gamma=gamma)
        p=solve_scenario_update(c,stochastic_inputs,weather)
        assert p['accepted'],p['reason']
        stochastic_initial_plans[label]=p
    display(pd.DataFrame([{'Controller':label,'First action (kW)':p['power_kw'][0],
        'Planned bill (£)':p['bill_gbp'],'Predicted penalty (£)':p['penalty_gbp']}
        for label,p in stochastic_initial_plans.items()]).round(4))
    print('PASS: single-scenario and repeated-scenario objective checks.')
    ''')

    code('stochastic-plan-figure', r"""
    fig,axes=plt.subplots(3,1,figsize=(12.5,7),sharex=True)
    interval_hours=np.arange(HORIZON)*dt;boundary_hours=np.arange(HORIZON+1)*dt
    axes[0].plot(interval_hours,outdoor_scenarios.T,color='tab:blue',alpha=.18,lw=.8)
    axes[0].plot(interval_hours,mean_weather[0],color='black',lw=1.8,label='Conditional mean forecast')
    axes[0].set_ylabel('Outdoor (°C)');axes[0].legend(fontsize=8)
    plan_colours={'Mean-forecast MPC':'tab:green','Scenario MPC':'tab:blue'}
    scenario_plan=stochastic_initial_plans['Scenario MPC']
    axes[1].plot(boundary_hours,scenario_plan['temperature_scenarios_c'].T,color='tab:blue',alpha=.18,lw=.8)
    for label,p in stochastic_initial_plans.items():
        axes[1].plot(boundary_hours,p['temperature_c'],color=plan_colours[label],label=label+' (mean prediction)')
        axes[2].stairs(p['power_kw'],boundary_hours,color=plan_colours[label],label=label)
    axes[1].axhline(20.,color='.4',ls='--',label='Endpoint comfort bounds')
    axes[1].axhline(23.,color='.4',ls='--')
    axes[1].set_ylabel('Predicted indoor (°C)');axes[1].legend(fontsize=8,ncol=2)
    axes[2].set(xlabel='Hours from the initial decision',ylabel='Electrical P (kW)',xlim=(0,HORIZON*dt))
    axes[2].set_xticks(np.arange(0,HORIZON*dt+1))
    axes[2].legend(fontsize=8)
    fig.suptitle('Controlled winter case | 40 weather scenarios, one common heating plan',fontsize=11)
    save_figure(fig,'F82-14','Initial conditional plans; unknown interval weather; sigma=2 C; rho=0.8','w82-stochastic-plan-figure')
    export_table(pd.DataFrame(outdoor_scenarios),'stochastic_initial_weather.csv')
    export_table(pd.DataFrame(sampled_errors),'stochastic_planning_errors.csv')
    """)
    md('stochastic-loop-text', r"""
    ### Evaluate the policies over independent realisations

    Keep the planning scenarios fixed. Draw 200 **new** complete weather-error
    paths using a separate seed and the same Gaussian law. These paths belong
    to the evaluator. Each controller receives only its current indoor reading
    and past weather, and recomputes the conditional forecast at each step.

    ```text
    for each independently sampled realisation:
        initialise both controllers at 20 °C
        for each 30-minute interval k:
            1. Read the controller's current indoor measurement.
            2. Forecast upcoming weather using the previously observed error.
            3. Optimise/check its mean-forecast or scenario-based plan.
            4. Select the first action, with the existing backup rule.
            5. Realise the same weather for both policies and advance each state.
            6. Observe the new temperatures and weather; repeat.
        record each controller's realised cost on this path
    ```

    Score the same quantity that the controller minimises:

    $$J^{(i)}=\Delta t\sum_{k=0}^{M-1}p_kP_k^{\mathrm{act},(i)}
       +\gamma\Delta t\sum_{k=0}^{M-1}\max\{L_k^{\mathrm{end}}-T_{k+1}^{(i)},0\},
    \qquad \widehat{\mathbb E}[J]=\frac1R\sum_{i=1}^{R}J^{(i)}.$$

    [i labels an evaluation realisation; r earlier labelled an optimisation
    scenario. The two samples use separate random draws.] Report bill and
    comfort separately as well. Record upper-bound violations separately because
    the optimisation treats that bound as hard. Both controllers see the same
    weather on path i, making their cost difference a paired comparison.

    The 95% Monte Carlo intervals below quantify sampling uncertainty in the
    **estimated means** under this fixed model and policy pair. They use
    mean ± 1.96 SD/√R. They describe a different quantity from the spread of
    individual operating costs or a comfort-probability guarantee.
    """)
    code('stochastic-operation', r"""
    rng_evaluation=np.random.default_rng(EVALUATION_SEED)
    evaluation_covariance=error_covariance(REPORTING_STEPS,SIGMA_WEATHER,RHO)
    realised_errors=rng_evaluation.multivariate_normal(
        np.zeros(REPORTING_STEPS),evaluation_covariance,size=N_REALISATIONS)
    stochastic_metrics,stochastic_intervals,stochastic_saved_plans=monte_carlo_mpc(
        model,make_scenario_controller,sampled_errors,realised_errors,rho=RHO,gamma=gamma)
    costs=stochastic_metrics.pivot(index='path',columns='strategy',values='total_cost_gbp')
    stochastic_pairs=pd.DataFrame({'mean_forecast_gbp':costs['Mean-forecast MPC'],
        'scenario_gbp':costs['Scenario MPC']})
    stochastic_pairs['delta_gbp']=stochastic_pairs.scenario_gbp-stochastic_pairs.mean_forecast_gbp
    differences=stochastic_pairs.delta_gbp
    delta_mean=differences.mean();delta_se=differences.std(ddof=1)/np.sqrt(N_REALISATIONS)
    stochastic_evidence={'realisations':N_REALISATIONS,'mean_delta_gbp':delta_mean,
        'mean_delta_ci_low':delta_mean-1.96*delta_se,'mean_delta_ci_high':delta_mean+1.96*delta_se,
        'fraction_lower_cost':float((differences<0).mean()),
        'mean_cost_reduction_percent':100*(1-costs['Scenario MPC'].mean()/costs['Mean-forecast MPC'].mean())}
    stochastic_summary=[]
    for label,group in stochastic_metrics.groupby('strategy',sort=False):
        row={'strategy':label,'realisations':len(group),'failed':int((~group.complete).sum()),
            'fallbacks':int(group.fallback_count.sum())}
        for metric in ['bill_gbp','penalty_gbp','total_cost_gbp','below_degree_hours','above_degree_hours']:
            values=group[metric]
            row[metric+'_mean']=values.mean()
            row[metric+'_se']=values.std(ddof=1)/np.sqrt(len(values))
        row['total_cost_p10']=group.total_cost_gbp.quantile(.1)
        row['total_cost_p90']=group.total_cost_gbp.quantile(.9)
        stochastic_summary.append(row)
    stochastic_summary=pd.DataFrame(stochastic_summary)
    display(stochastic_summary[['strategy','bill_gbp_mean','penalty_gbp_mean','total_cost_gbp_mean',
        'below_degree_hours_mean','above_degree_hours_mean','failed','fallbacks']].round(4))
    display(pd.Series(stochastic_evidence).round(4))
    export_table(stochastic_metrics,'stochastic_mc_metrics.csv')
    export_table(stochastic_intervals,'stochastic_mc_intervals.csv')
    export_table(stochastic_pairs.reset_index(),'stochastic_paired_costs.csv')
    export_table(stochastic_summary,'stochastic_summary.csv')
    export_table(pd.DataFrame(realised_errors),'stochastic_evaluation_errors.csv')
    export_json(stochastic_evidence,'stochastic_monte_carlo_evidence.json')
    """)
    code('stochastic-performance-figure', r"""
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    common_bins=np.linspace(stochastic_metrics.total_cost_gbp.min(),stochastic_metrics.total_cost_gbp.max(),23)
    for label,group in stochastic_metrics.groupby('strategy',sort=False):
        ax=axes[0];colour=plan_colours[label]
        ax.hist(group.total_cost_gbp,bins=common_bins,weights=np.full(len(group),1/len(group)),
            alpha=.45,color=colour,label=label)
        ax.axvline(group.total_cost_gbp.mean(),color=colour,ls='--',lw=1.6)
    axes[0].set(xlabel='Realised total cost J (£)',ylabel='Fraction of realisations')
    axes[0].legend(fontsize=8)
    axes[1].hist(differences,bins=23,weights=np.full(N_REALISATIONS,1/N_REALISATIONS),color='tab:blue',alpha=.7)
    axes[1].axvline(0,color='black',lw=1,label='Equal realised cost')
    axes[1].axvline(delta_mean,color='tab:red',ls='--',label=f'Mean difference {delta_mean:.2f} £')
    axes[1].set(xlabel='Paired cost: scenario − mean-forecast (£)',ylabel='Fraction of realisations')
    axes[1].legend(fontsize=8)
    fig.suptitle('200 independent weather paths | cost = bill + 25 × lower-comfort degree-hours',fontsize=11)
    save_figure(fig,'F82-15','Independent Monte Carlo distributions and paired total-cost differences','w82-stochastic-performance-figure')
    """)
    code('stochastic-mean-figure', r"""
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for i,row in enumerate(stochastic_summary.itertuples()):
        axes[0].bar(i,row.bill_gbp_mean,color='.65',width=.55,label='Electricity bill' if i==0 else None)
        axes[0].bar(i,row.penalty_gbp_mean,bottom=row.bill_gbp_mean,color='tab:orange',width=.55,
            label='Lower-comfort penalty' if i==0 else None)
        axes[0].errorbar(i,row.total_cost_gbp_mean,yerr=1.96*row.total_cost_gbp_se,fmt='none',color='black',capsize=4)
        axes[0].annotate(f'{row.total_cost_gbp_mean:.2f}',(i,row.total_cost_gbp_mean+1.96*row.total_cost_gbp_se),
            xytext=(0,6),textcoords='offset points',ha='center')
        axes[1].bar(i,row.below_degree_hours_mean,color=plan_colours[row.strategy],width=.55)
        axes[1].errorbar(i,row.below_degree_hours_mean,yerr=1.96*row.below_degree_hours_se,
            fmt='none',color='black',capsize=4)
    for ax in axes:
        ax.set_xticks([0,1],['Mean-forecast MPC','Scenario MPC']);ax.grid(axis='x',visible=False)
        ax.set_ylim(0,ax.get_ylim()[1]*1.15)
    axes[0].set_ylabel('Estimated expected total cost (£)');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Mean true lower discomfort (°C h)')
    fig.suptitle('Means over 200 paths | error bars = approximate 95% Monte Carlo intervals for each mean',fontsize=10)
    save_figure(fig,'F82-16','Monte Carlo means: bill/comfort cost decomposition and lower discomfort','w82-stochastic-mean-figure')
    """)
    md('stochastic-interpretation', r"""
    **Interpret:** compare the two cost distributions and the paired differences.
    Is the estimated mean difference below zero? Does its Monte Carlo interval
    support the same conclusion? Separate the electricity bill from the comfort
    penalty and report any upper-bound violations or backup use.

    Explain why a controller may spend more on heating while reducing expected
    total cost. Compare the uncertain first interval with the core's exact-current-
    weather assumption. Scenario MPC's later actions remain provisional and are
    recomputed after each observation: this is a receding-horizon evaluation of
    both policies, with one common plan across scenarios at each update.

    This controlled example uses a correctly specified Gaussian error law,
    exact sensing and a known physical model. The 200 paths estimate performance
    under these assumptions. A different error law, comfort value or actuation
    limit can change the ranking. Broader recommendations require suitable
    independent data and evaluation. The sampled hard upper constraints are
    checked on planning paths; comfort-probability guarantees require further analysis.
    """)
    md('response-stochastic', r"""
    Use the computed evidence: report the estimated expected total costs, their
    paired difference and Monte Carlo interval. The scenario policy accounts for
    adverse weather before choosing the heating action. The mean-forecast policy
    predicts the conditional mean but can incur lower-comfort shortfalls when
    the newly realised weather is colder. Additional heating can reduce those
    penalties enough to lower the expected bill-plus-comfort cost. Interpret
    individual paths separately from the mean over realisations.

    Planning scenarios and evaluation paths use separate random draws. The
    fixed protocol was checked on development paths with seed 812; the 200
    evaluation paths use seed 8282. The result concerns this declared stress
    case and the chosen cost function.
    """ if solution else '*Write your interpretation here.*')
    return cells
