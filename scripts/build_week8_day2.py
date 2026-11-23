"""Author paired notebooks for isolated error experiments around the Day 1 MPC."""
from pathlib import Path
import textwrap,csv,sys
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.week8_day2_stochastic_cells import extension_cells,EXTENSION_TASKS,EXTENSION_FIGURES
STEM='tutorial_8_2_mpc_uncertainty_evaluation'
TASKS=[('A',6,12),('B',12,12),('C',15,8),('D',20,12),('E',26,10),('F',29,10)]
SLIDE_MINUTES=[2, 3, 6, 6, 7, 12, 7, 7, 5, 5, 5, 12, 6, 5, 8, 4, 5, 5, 5, 12, 7, 6, 5, 5, 5, 10, 4, 4, 10, 7, 0, 2]
FIGURES=[('F82-01','error_cases'),('F82-02','forecast_errors'),('F82-03','model_error'),
 ('F82-04','sensor_process_errors'),('F82-05','combined_errors'),('F82-06','case_metrics'),
 ('F82-07','held_out_pairs'),('F82-08','richer_reference'),('F82-09','case_table'),('F82-10','forecast_consequences'),
 ('F82-12','sensor_detail'),('F82-13','forecast_policy_bars')]+EXTENSION_FIGURES

def make_notebook(solution=False):
    cells=[]
    def md(key,text):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip(),id='w82-'+key))
    def code(key,text,tags=None):cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip(),id='w82-'+key,metadata={'tags':tags or ['supplied']}))
    def task(letter,answer,scaffold,kind='code'):
        ctor=nbf.v4.new_markdown_cell if kind=='markdown' else nbf.v4.new_code_cell
        cells.append(ctor(textwrap.dedent(answer if solution else scaffold).strip(),id='w82-task-'+letter.lower(),metadata={'tags':['task',letter]}))
    def response(letter,text):md('response-'+letter.lower(),text if solution else '*Write your interpretation here.*')
    md('purpose',r'''
    # Tutorial 8.2: How errors affect building MPC

    Tutorial 8.1 implemented MPC and compared it with a fixed open-loop plan and
    the midpoint thermostat on a richer synthetic reference. Several errors acted
    together. We now keep that controller and separate their effects: imperfect
    forecast noise, forecast bias, model mismatch, sensor errors and physical
    disturbances, then their combination. We compare fixed and MPC operation
    using costs, demand and true comfort. C0 is a short no-error reference check;
    the forecast experiments focus on C1 noise versus C2 bias.

    | Block | Slides | Minutes | Focus |
    |---|---|---:|---|
    | 1 | 1–8 | 50 | Error formulations and experimental cases; Task A |
    | 2 | 9–16 | 50 | Forecast noise/bias, then model error; Tasks B–C |
    | Break | after 16 | 15–20 additional | Outside the teaching total |
    | 3 | 17–24 | 50 | Sensor/process errors and combined effects; Task D |
    | 4 | 25–32 | 42 | Operating metrics and held-out evidence; Tasks E–F |

    The six tasks take 12,12,8,12,10,10 minutes including debriefs: 64 within 192
    teaching minutes. A further 32 minutes of guided evidence and interpretation
    bring active tutorial time to 96 minutes within these blocks. Preparation,
    simulation, acceptance checks, scoring and plotting are supplied. Run this
    notebook independently from the repository root, starting with a fresh kernel.

    A separate **40-minute extension** introduces stochastic MPC using Week 4's
    joint Gaussian error vectors. Tasks G–H construct weather scenarios and
    one shared heating plan, followed by an independent Monte Carlo evaluation.
    ''')
    code('setup',r'''
    from pathlib import Path
    from dataclasses import asdict
    import json,hashlib,time,platform
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from IPython.display import display
    from scripts.week8_review_diagnostics import case_evidence
    from course_utils.week8_control import run_experiment,realised_metrics,solver_configuration
    from course_utils.week8_errors import (ErrorCase,CASES,CORE_CASES,innovations,forecast_table,
        run_case,batch,pairs,sources,freeze_protocol,verify_protocol)
    ROOT=Path.cwd()
    DATA_DIR=ROOT/'data/reference_building'
    if not (DATA_DIR/'operation_conditions.csv').is_file():raise FileNotFoundError('Run from repository root.')
    EXPORT_ARTEFACTS=True
    OUTPUT_DIR=ROOT/'teaching/week8_day2_outputs'
    FIGURE_DIR=ROOT/'figures/week8_day2'
    WORKED_DAY='2025-02-03'
    DEVELOPMENT_DATES=[WORKED_DAY]
    FINAL_DATES=['2025-02-09','2025-02-10','2025-02-11']
    SEEDS=[11,22,33]
    N=M=48
    dt=.5
    gamma=25.
    comfort_margin_c=0.
    SOLVER,solver_options=solver_configuration()
    operation=pd.read_csv(DATA_DIR/'operation_conditions.csv',index_col='timestamp',parse_dates=True)
    supplied_forecasts=pd.read_csv(DATA_DIR/'operation_forecasts.csv',parse_dates=['issue_time','valid_time'])
    model=json.loads((DATA_DIR/'verified_controller_model.json').read_text(encoding='utf8'))
    COLOURS={'Fixed schedule':'tab:green','MPC':'tab:blue','Thermostat':'tab:red'}
    plt.rcParams.update({'font.size':11,'figure.dpi':110,'axes.grid':True,'grid.alpha':.2,'svg.fonttype':'none'})
    print('Unchanged Day 1 controller |',SOLVER,'| N=M=48')
    ''')
    md('formulations',r'''
    ## 1. Keep the controller; change one error source

    Retain the prediction model and lower-soft/upper-hard optimisation from 8.1:

    The first index gives the absolute target time; the index after the bar
    gives the planning time $k$. Write $\tau=k+j$ for the absolute target time,
    where $j$ is the local array offset. All entries in a plan are calculated
    using the information available at $k$. The hat marks model predictions and
    weather forecasts. Measured and simulated true temperatures retain their
    separate superscripts.

    $$\widehat T_{\tau+1|k}=f_\theta(\widehat T_{\tau|k},P_{\tau|k},\widehat T_{o,\tau|k})
    =(1-b)\widehat T_{\tau|k}+b\widehat T_{o,\tau|k}+c\,\mathrm{COP}\,P_{\tau|k},$$

    $$\widehat T_{k|k}=T_k^{\mathrm{meas}},\qquad
    \min_{P,\widehat T,s}\Delta t\sum_{\tau=k}^{k+N-1}[p_{\tau}P_{\tau|k}+\gamma s_{\tau|k}],$$

    $$0\le P_{\tau|k}\le5,\ s_{\tau|k}\ge0,\quad
    \widehat T_{\tau+1|k}+s_{\tau|k}\ge L^{\mathrm{end}}_\tau+\delta,\quad
    \widehat T_{\tau+1|k}\le U^{\mathrm{end}}_\tau-\delta.$$

    These constraints hold for $\tau=k,\ldots,k+N-1$.
    $L^{\mathrm{end}}_\tau,U^{\mathrm{end}}_\tau$ apply to the end of interval
    $\tau$, at boundary $\tau+1$. The CSV profiles and code arrays retain their
    existing names: `lower[j]`, `upper[j]` constrain the prediction at `T[j+1]`.
    Announced prices and comfort schedules, current readings and future weather
    forecasts are available at $k$.

    The supplied values remain $b=0.0234549361$, $c=0.1261192467$, COP=3,
    $\Delta t=0.5$ h, $N=M=48$, $\gamma=25$ £/(°C h), and core
    $\delta=0$. MPC applies only the first accepted action and updates
    every 30 minutes. The fixed policy commits to the same initial 48-action
    problem. The thermostat's different temperature preference was examined in
    8.1; we focus on fixed/MPC here to keep the error comparison interpretable.

    For an accepted MPC plan, $P_k^{\mathrm{act}}=P^*_{k|k}$ is the applied
    first action; later $P^*_{\tau|k}$ values remain provisional. The supplied
    backup policy handles rejected plans or unavailable readings, and tiny
    numerical corrections at actuator limits follow the acceptance tolerance.

    Abstract the reference dynamics and available information as

    $$T^{true}_{k+1}=f_{\theta_{ref,k}}(T^{true}_k,P_k^{\mathrm{act}},T_{o,k})+w_{k+1},\tag{R}$$

    $$T_k^{meas}=T_k^{true}+\beta_y+\nu_k,\tag{S}$$

    $$\widehat T_{o,\tau|k}=T_{o,k+j}+\beta_f+\eta_{k,j},\quad j\ge1.\tag{F}$$

    The disturbance $w_{k+1}$ acts during the transition from $k$ to $k+1$.
    The next reading $T_{k+1}^{\mathrm{meas}}$ becomes available after applying
    the current action. Process and sensor noise are latent simulation inputs;
    the controller receives the reading. In (F), $\eta_{k,j}$ is the injected
    forecast perturbation for the plan issued at $k$ and lead $j$; its
    mathematical labels distinguish issue time from lead time.

    Model mismatch changes $\theta_{ref,k}$ in (R). Process noise
    $w$ changes the physical state; sensor noise $\nu$
    and bias $\beta_y$ change its reported value. Forecast noise
    $\eta$ and bias $\beta_f$ perturb the future information
    provided to the controller. Actual weather stays fixed. Price is a known
    economic input and comfort bounds are imposed requirements. The controller
    coefficients $\theta$ stay fixed.

    For independent tests, the **controlled reference** uses the same one-state
    recurrence as the controller, starting at 20°C. **C0 No-error reference** has
    $\theta_{ref}=\theta$, $w=\nu=\eta=0$ and $\beta_y=\beta_f=0$. The evaluator
    constructs exact future-weather arrays as an **oracle diagnostic** for
    evaluating the controller under exact weather information. Other cases receive
    explicitly perturbed arrays. The current outdoor nowcast ($j=0$)
    remains exact. The controller receives the explicitly supplied forecast
    arrays.

    C0 therefore has exact forecasts, exact measurements, zero process noise
    and reference coefficients equal to the controller coefficients. It checks
    the implementation and supplies the reference cost/comfort values. On this
    supplied day, fixed and MPC give the same result to solver precision. We
    use the error cases to examine how feedback changes operation.

    Keeping 8.1's richer reference would leave structural mismatch active even in
    a forecast-error-only experiment. The controlled one-state reference removes that
    confounding cause. At the end we return separately to the unchanged richer
    reference and original forecast CSV, connecting back to 8.1.
    ''')
    md('case-definitions',r'''
    The error magnitudes are illustrative teaching settings. All unspecified
    channels are zero and $c_{ref}=c$, COP=3 throughout.

    Red terms in the equations below identify the changed quantity. Each term
    also has a written label. These are controlled error injections for teaching.

    | Case | Only change from C0 No-error reference |
    |---|---|
    | C0 No-error reference | Exact forecasts/sensing, matching coefficients, zero process noise |
    | C1 Forecast noise | $\eta_{k,j}\sim\mathcal N(0,1^2)$ °C at future leads |
    | C2 Forecast bias | $\beta_f=+1.5$°C at every future origin/lead |
    | C3 Model error | $b_{ref}=1.3b=0.0304914170$; controller $b$ unchanged |
    | C4 Sensor noise | $\nu_k\sim\mathcal N(0,0.2^2)$ °C |
    | C5 Sensor bias | $\beta_y=+0.4$°C |
    | C6 Process noise | $w_{k+1}\sim\mathcal N(0,0.05^2)$ °C per transition |
    | C7 Combined errors | All C1–C6 changes together |
    | C8 Midday model drift | $b_{ref}=b$ before 12:00, $1.3b$ afterwards; other channels exact |

    Forecast error retains Week 4.2's actual-minus-forecast sign. A positive warm
    forecast perturbation therefore gives a negative forecast error. A finite
    sample of zero-mean noise can have a nonzero average. C1 uses independent
    noise at each origin and lead; temporally dependent forecast errors would
    require a different sampling model.

    C1/C4/C6/C7 use seeds 11,22,33. Each deterministic case has one run.
    Stochastic cases use the three seeds to describe variation in the outcomes.
    Separate process, sensor and forecast arrays are generated before a simulation
    and reused across policies at matching times. Different actions can still lead
    to different true temperatures and readings under the same innovations.
    ''')
    md('task-a-text','### Task A — locate each error (slide 6; 12 min)\n\nUse 8 minutes to match the changes to (R), (S) or (F), then 4 minutes to explain what is exact in C0 and why we change one error source at a time. Distinguish changed physics, information and coefficients.')
    task('A',r'''
    | Change | Equation | Direct effect |
    |---|---|---|
    | Forecast noise/bias | (F) | Future information; actual weather fixed |
    | Model mismatch/drift | (R) | Physical response; controller model fixed |
    | Sensor noise/bias | (S) | Available reading; no direct physical change |
    | Process noise | (R) | Physical state after transition |

    C1 retains identical reference/controller coefficients, exact sensors and zero process noise.
    ''',r'''
    | Change | Equation | Direct effect |
    |---|---|---|
    | Forecast noise/bias | TODO | TODO |
    | Model mismatch/drift | TODO | TODO |
    | Sensor noise/bias | TODO | TODO |
    | Process noise | TODO | TODO |

    TODO: explain why C1 isolates forecast error here but 8.1 contained several errors.
    ''','markdown')
    code('helpers','FIGURE_SPECS='+repr(FIGURES)+r'''
figure_manifest=[]
figure_provenance={
 'F82-01':dict(seeds='Not applicable: settings',dates=WORKED_DAY,csv='case_settings_table.csv',units='degrees C;dimensionless'),
 'F82-02':dict(seeds='11',dates=WORKED_DAY,csv='worked_runs/*_C[12]_11_MPC_forecast_records.csv',units='degrees C'),
 'F82-03':dict(seeds='11 (deterministic)',dates=WORKED_DAY,csv='worked_runs/*_C[038]_11_*_intervals.csv;worked_runs/*_C[038]_11_*_boundaries.csv',units='degrees C;kW'),
 'F82-04':dict(seeds='11',dates=WORKED_DAY,csv='worked_runs/*_C[456]_11_MPC_*.csv',units='degrees C;kW'),
 'F82-05':dict(seeds='11',dates=WORKED_DAY,csv='worked_runs/*_C7_11_*.csv',units='degrees C;kW'),
 'F82-06':dict(seeds='11;22;33 stochastic; 11 deterministic',dates=WORKED_DAY,csv='worked_summary.csv',units='GBP;degrees C h'),
 'F82-07':dict(seeds='C1/C7:11;22;33; C2/C3:11',dates=';'.join(FINAL_DATES),csv='final_paired_differences.csv;final_case_evidence.csv;held_out_bar_summary.csv',units='GBP;degrees C h'),
 'F82-08':dict(seeds='90',dates=FINAL_DATES[0],csv='transfer_runs/*.csv;richer_reference_transfer_metrics.csv',units='degrees C;kW'),
 'F82-09':dict(seeds='11;22;33 stochastic; 11 deterministic',dates=WORKED_DAY,csv='worked_operating_table.csv',units='GBP;degrees C h'),
 'F82-10':dict(seeds='11; C1 is one noise realisation, C2 deterministic',dates=WORKED_DAY,csv='w8d2_F10_forecast_consequences.csv',units='degrees C;kW'),
 'F82-12':dict(seeds='C4/C6:11;22;33; C5:11',dates=WORKED_DAY,csv='w8d2_F12_sensor_detail.csv',units='kWh;GBP;kW;degrees C h'),
 'F82-13':dict(seeds='C1:11;22;33; C2:11',dates=WORKED_DAY,csv='forecast_only_summary.csv;forecast_policy_pairs.csv',units='GBP;degrees C h'),
 'F82-14':dict(seeds='811; 40 joint weather scenarios',dates='Controlled 6-hour winter case',csv='stochastic_initial_weather.csv;stochastic_planning_errors.csv',units='degrees C;kW'),
 'F82-15':dict(seeds='8282; 200 independent weather paths',dates='Controlled 6-hour winter case',csv='stochastic_mc_metrics.csv;stochastic_paired_costs.csv',units='GBP'),
 'F82-16':dict(seeds='8282; 200 independent weather paths',dates='Controlled 6-hour winter case',csv='stochastic_mc_metrics.csv;stochastic_summary.csv',units='GBP;degrees C h')}

def export_table(frame,name):
    if EXPORT_ARTEFACTS:
        OUTPUT_DIR.mkdir(parents=True,exist_ok=True);frame.to_csv(OUTPUT_DIR/name,index=False)
def export_json(value,name):
    if EXPORT_ARTEFACTS:
        OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
        (OUTPUT_DIR/name).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
def save_figure(fig,id,case,producer):
    stem=next(stem for key,stem in FIGURE_SPECS if key==id)
    stochastic=id in {'F82-14','F82-15','F82-16'}
    fig.tight_layout(rect=(0,0,1,.94))
    if EXPORT_ARTEFACTS:
        FIGURE_DIR.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg']:fig.savefig(FIGURE_DIR/f'w8d2_{id.replace("F82-","F")}_{stem}.{ext}',dpi=220)
        figure_manifest.append({'id':id,'case':case,'cell_tag':producer,
            'N':HORIZON if stochastic else N,'M':REPORTING_STEPS if stochastic else M,'gamma':gamma,
            'margin_c':0.,'seeds':figure_provenance[id]['seeds'],
            'dates':figure_provenance[id]['dates'],'csv_evidence':figure_provenance[id]['csv'],
            'units':figure_provenance[id]['units'],'notebook':'tutorial_8_2_mpc_uncertainty_evaluation_solution.ipynb',
            'reference':'Richer two-state' if id=='F82-08' else 'Controlled one-state',
            'png':f'figures/week8_day2/w8d2_{id.replace("F82-","F")}_{stem}.png',
            'svg':f'figures/week8_day2/w8d2_{id.replace("F82-","F")}_{stem}.svg',
            'source_sha256':json.dumps({**sources(ROOT),**{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['scripts/week8_review_diagnostics.py','course_utils/week8_stochastic.py','scripts/week8_day2_stochastic_cells.py']}},sort_keys=True),'completion_status':'complete',
            'caption':case,'information':('Current indoor reading and past weather; unknown upcoming interval weather; independent Monte Carlo paths.'
                if stochastic else 'Explicit future forecast table; current nowcast; measured initial state; original-bound true scoring.')})
    plt.show();plt.close(fig)
def table_plot(frame,id,case,height=3.5,producer=None):
    fig,ax=plt.subplots(figsize=(12.5,height));ax.axis('off')
    fig.suptitle(case,fontsize=11)
    tab=ax.table(cellText=frame.values,colLabels=frame.columns,loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.7)
    for (row,col),cell in tab.get_celld().items():
        cell.set_edgecolor('.8')
        if row==0:cell.set_facecolor('.92')
    save_figure(fig,id,case,producer or {'F82-01':'w82-worked-batch','F82-09':'w82-summary-figures','F82-12':'w82-sensor-detail'}[id])
def clock(ax,day=WORKED_DAY):
    start=pd.Timestamp(day);ax.set_xlim(start,start+pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,4)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
def comfort(ax,rows):
    ax.plot(rows.interval_end,rows.lower_c,'.--',color='.4',ms=2,label='Original bounds')
    ax.plot(rows.interval_end,rows.upper_c,'.--',color='.4',ms=2)
def result_at(case,strategy='MPC',seed=11):return worked_runs[(WORKED_DAY,case,seed,strategy)]
def summarise(metrics):
    records=[]
    for (case,strategy),group in metrics.groupby(['case','strategy'],sort=False):
        valid=group.loc[group.complete];row={'case':case,'strategy':strategy,'valid':len(valid),'failed':len(group)-len(valid)}
        for col in ['energy_kwh','bill_gbp','peak_kw','below_degree_hours','above_degree_hours','total_degree_hours']:
            row[col+'_mean']=valid[col].mean();row[col+'_min']=valid[col].min();row[col+'_max']=valid[col].max()
        records.append(row)
    return pd.DataFrame(records)
def export_runs(runs,prefix):
    for (day,case,seed,strategy),result in runs.items():
        stem=f"{prefix}/{day}_{case.split()[0]}_{seed}_{strategy.replace(' ','_')}"
        if EXPORT_ARTEFACTS:(OUTPUT_DIR/prefix).mkdir(parents=True,exist_ok=True)
        for field in ['intervals','boundaries','forecast_records']:export_table(result[field],stem+'_'+field+'.csv')
''')
    code('worked-batch',r'''
    started=time.perf_counter()
    worked_metrics,worked_runs=batch(operation,model,list(CASES),DEVELOPMENT_DATES,SEEDS)
    worked_runtime_s=time.perf_counter()-started
    case_table=pd.DataFrame([{'Case':name.split()[0],'Forecast σ / bias':f'{case.forecast_std_c:g} / {case.forecast_bias_c:g}',
        'b_ref / b':('1 → 1.3 at noon' if case.drift_after_step else f'{case.loss_multiplier:g}'),
        'Sensor σ / bias':f'{case.sensor_std_c:g} / {case.sensor_bias_c:g}',
        'Process σ':f'{case.process_std_c:g}'} for name,case in CASES.items()])
    display(case_table);table_plot(case_table,'F82-01','Error-channel settings; noise/bias in °C',4.5)
    export_table(case_table,'case_settings_table.csv');export_table(worked_metrics,'worked_metrics.csv')
    export_json({name:asdict(case) for name,case in CASES.items()},'case_definitions.json')
    print('Worked runs:',len(worked_metrics),'| completed:',int(worked_metrics.complete.sum()),'| seconds:',round(worked_runtime_s,2))
    ''')
    md('forecast-text',r'''
    ## 2. Forecast noise, then forecast bias (slides 9–13)

    **C1 Forecast noise** adds a new perturbation at each issue time and future
    lead. For $\tau=k+j$, $j\ge1$,

    $$\widehat T_{o,\tau|k}=T_{o,\tau}
      +\underbrace{\color{red}{\eta_{k,j}}}_{\text{forecast noise}},\qquad
      \eta_{k,j}\sim\mathcal N(0,1^2)\ {^\circ\mathrm C}.$$

    **C2 Forecast bias** shifts every future forecast by the same warm offset:

    $$\widehat T_{o,\tau|k}=T_{o,\tau}
      +\underbrace{\color{red}{\beta_f}}_{\text{persistent forecast bias}},\qquad
      \beta_f=+1.5\ {^\circ\mathrm C},\quad \tau>k.$$

    Both cases retain the exact current outdoor input,
    $\widehat T_{o,k|k}=T_{o,k}$. Actual weather, reference dynamics and exact
    measurements stay fixed; process noise is zero. The fixed policy commits to
    the midnight plan. MPC replaces future forecasts and replans every 30 minutes
    using the new measurement. C2's warm bias persists at each update. The two
    error magnitudes illustrate different mechanisms; they are separate settings.

    ### Task B — compare forecast cases (slide 12; 12 min)

    Select C1/C2 into `forecast_metrics`, use `summarise` for bill, true discomfort
    and valid/failed counts, then form `forecast_pairs=pairs(forecast_metrics)`.
    Compare MPC with fixed within each error case. Use 8 minutes to interpret
    the table and 4 for debrief. C1 has three seeds; C2 has one deterministic run.
    ''')
    task('B',r'''
    forecast_cases=['C1 Forecast noise','C2 Forecast bias']
    forecast_metrics=worked_metrics.loc[worked_metrics['case'].isin(forecast_cases)]
    forecast_summary=summarise(forecast_metrics)
    forecast_pairs=pairs(forecast_metrics)
    display(forecast_summary[['case','strategy','valid','failed','bill_gbp_mean','total_degree_hours_mean']].round(4))
    display(forecast_pairs[['case','seed','valid_pair','delta_bill_gbp','delta_total_degree_hours']].round(4))
    ''',r'''
    # TODO B: select C1/C2 into forecast_metrics and summarise it in forecast_summary.
    # Define forecast_pairs=pairs(forecast_metrics); display cost, discomfort and run counts.
    raise NotImplementedError('Task B: compare MPC and fixed under forecast noise and bias.')
    ''')
    code('forecast-figure',r'''
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5));origin=pd.Timestamp(WORKED_DAY)
    valid=pd.date_range(origin,periods=N,freq='30min')
    axes[0].plot(valid,operation.loc[valid,'outdoor_temperature_c'],color='black',lw=2,label='Realised outdoor T')
    for name,colour in zip(['C1 Forecast noise','C2 Forecast bias'],['tab:orange','tab:purple']):
        records=result_at(name)['forecast_records'];first=records.loc[records.issue_time==origin]
        axes[0].plot(first.valid_time,first.outdoor_input_c,ls='--',color=colour,label=name)
        axes[1].plot(first.valid_time,first.actual_minus_forecast_c,color=colour,label=name)
    for ax in axes:clock(ax);ax.set_xlabel('Valid time');ax.legend(fontsize=8)
    axes[0].set_ylabel('Outdoor temperature (°C)');axes[1].set_ylabel('Actual − forecast (°C)');axes[1].axhline(0,color='black',lw=.7)
    fig.suptitle('3 February | controlled reference | first-origin forecast | seed 11',fontsize=11)
    save_figure(fig,'F82-02','C1 noise versus C2 bias: first-origin forecasts, with an exact current input','w82-forecast-figure')
    export_table(forecast_summary,'forecast_only_summary.csv')
    export_table(forecast_pairs,'forecast_policy_pairs.csv')
    ''')
    response('B',r'''
    In C1, MPC has less true discomfort than fixed for each of the three supplied
    seeds. The bill difference changes sign across seeds. Replanning corrects
    actions after new readings and forecasts arrive, while the fixed policy
    retains its first noisy plan. In C2, the fixed plan has a lower bill and much
    more discomfort; its warm forecasts understate heating needs. MPC revises
    actions from the exact state reading while retaining biased future forecasts.
    These outcomes describe the supplied day and settings; compare cost and
    comfort together.
    ''')
    code('forecast-consequences',r"""
    fig,axes=plt.subplots(2,2,figsize=(12.5,6));forecast_consequences=[]
    for col,name in enumerate(['C1 Forecast noise','C2 Forecast bias']):
        for strategy in ['Fixed schedule','MPC']:
            result=result_at(name,strategy);rows=result['intervals'];states=result['boundaries']
            axes[0,col].plot(states.state_time,states.true_temperature_c,color=COLOURS[strategy],label=strategy)
            axes[1,col].stairs(rows.applied_power_kw,states.state_time,color=COLOURS[strategy],label=strategy)
            data=rows[['interval_start','interval_end','true_end_c','applied_power_kw','lower_c','upper_c']].copy()
            data['true_start_c']=states.true_temperature_c.iloc[:-1].to_numpy()
            data['case']=name;data['strategy']=strategy;data['seed']=11;forecast_consequences.append(data)
        comfort(axes[0,col],rows);axes[0,col].set_title(name)
        axes[0,col].set_ylabel('Simulated true T (°C)');axes[1,col].set_ylabel('Applied electrical P (kW)')
        for ax in axes[:,col]:clock(ax);ax.set_xlabel('Time');ax.legend(fontsize=8)
    fig.suptitle('3 February | identical weather and reference coefficients | seed 11',fontsize=11)
    save_figure(fig,'F82-10','C1/C2: fixed and MPC operation under forecast noise and persistent bias','w82-forecast-consequences')
    export_table(pd.concat(forecast_consequences,ignore_index=True),'w8d2_F10_forecast_consequences.csv')
    """)
    md('forecast-consequences-text',r'''
    Compare fixed and MPC within each column (slide 13). C1 shows one noise
    realisation, seed 11; the bars below retain all three C1 seeds. C2 shows the
    persistent warm bias. Relate the heating actions to true comfort and the
    paired bill differences. The panels use separate vertical scales for the
    two error cases so that the smaller C1 changes remain visible.
    ''')

    code('forecast-policy-bars',r'''
    fig,axes=plt.subplots(2,2,figsize=(12.5,6))
    strategies=['Fixed schedule','MPC'];x=np.arange(2)
    for row,name in enumerate(['C1 Forecast noise','C2 Forecast bias']):
        frame=forecast_summary.loc[forecast_summary['case']==name].set_index('strategy').loc[strategies]
        for col,(metric,unit) in enumerate([('bill_gbp','HP bill (£)'),('total_degree_hours','True discomfort (°C h)')]):
            ax=axes[row,col];means=frame[metric+'_mean'].to_numpy()
            lo=means-frame[metric+'_min'].to_numpy();hi=frame[metric+'_max'].to_numpy()-means
            bars=ax.bar(x,means,color=[COLOURS[s] for s in strategies],width=.55)
            repeated=frame.valid.to_numpy()>1
            ax.errorbar(x[repeated],means[repeated],yerr=np.vstack([lo[repeated],hi[repeated]]),fmt='none',color='black',capsize=5)
            for position,mean,upper in zip(x,means,means+hi):
                ax.annotate(f'{mean:.3f}',(position,upper),xytext=(0,4),textcoords='offset points',ha='center',va='bottom',fontsize=9)
            ax.set_xticks(x,[f'{s}\nn={int(n)}' for s,n in zip(strategies,frame.valid)])
            ax.set_ylabel(unit);ax.set_title(name,fontsize=11);ax.grid(axis='x',visible=False)
            ax.set_ylim(0,max(float(np.max(means+hi))*1.2,.01))
    fig.suptitle('3 February | bars = mean; whiskers = observed min–max across seeds | separate case scales',fontsize=10)
    save_figure(fig,'F82-13','C1: three noise seeds; C2: one deterministic run; ranges describe these runs','w82-forecast-policy-bars')
    ''')

    md('model-text',r'''
    ## 3. Model mismatch and drift (slides 14–16)

    **C3 Model error** changes the physical loss coefficient:

    $$T^{\mathrm{true}}_{k+1}
      =(1-{\color{red}1.3b})T^{\mathrm{true}}_k
       +{\color{red}1.3b}\,T_{o,k}+c\,\mathrm{COP}\,P_k^{\mathrm{act}}.$$

    The red coefficient gives 30% larger heat loss in the reference. The
    controller retains $b,c$; forecasts and measurements are exact.

    **C8 Midday model drift** changes that same coefficient at noon:

    $$T^{\mathrm{true}}_{k+1}=(1-b_{ref,k})T^{\mathrm{true}}_k
      +b_{ref,k}T_{o,k}+c\,\mathrm{COP}\,P_k^{\mathrm{act}},\qquad
      b_{ref,k}=\begin{cases}b,&k<24,\\{\color{red}1.3b},&k\ge24.\end{cases}$$

    The first changed transition is 12:00–12:30, so its first affected reading
    arrives at 12:30. Before that transition C8 agrees with C0.

    ### Task C — interpret model error (slide 15; 8 min)

    Select C0/C3/C8 and form a paired table using `pairs`. Inspect when the drift
    trace first diverges. Use 5 minutes for analysis and 3 for discussion. Explain
    what changes in the reference and what the controller keeps fixed.
    ''')
    task('C',r'''
    model_metrics=worked_metrics.loc[worked_metrics['case'].isin(['C0 No-error reference','C3 Model error','C8 Midday model drift'])]
    model_pairs=pairs(model_metrics)
    display(model_pairs[['case','valid_pair','delta_bill_gbp','delta_total_degree_hours']].round(4))
    ''',r'''
    # TODO C: select C0/C3/C8 into model_metrics and define model_pairs=pairs(model_metrics).
    # Display valid-pair flags and bill/discomfort differences.
    raise NotImplementedError('Task C: interpret isolated model error and midday drift.')
    ''')
    code('model-figure',r'''
    fig,axes=plt.subplots(2,2,figsize=(12.5,6))
    for col,name in enumerate(['C3 Model error','C8 Midday model drift']):
        for strategy in ['Fixed schedule','MPC']:
            result=result_at(name,strategy);base=result_at('C0 No-error reference',strategy)
            states=result['boundaries'];rows=result['intervals']
            axes[0,col].plot(base['boundaries'].state_time,base['boundaries'].true_temperature_c,color=COLOURS[strategy],ls=':',alpha=.7)
            axes[0,col].plot(states.state_time,states.true_temperature_c,color=COLOURS[strategy],label=strategy)
            axes[1,col].stairs(rows.applied_power_kw,states.state_time,color=COLOURS[strategy],label=strategy)
        comfort(axes[0,col],rows);axes[0,col].set_title(name+'; dotted = no-error reference',fontsize=10)
        for ax in axes[:,col]:clock(ax);ax.legend(fontsize=8);ax.set_xlabel('Time')
        if col==1:
            for ax in axes[:,col]:ax.axvline(pd.Timestamp(WORKED_DAY)+pd.Timedelta(hours=12),color='black',ls='--')
    for ax in axes[0]:ax.set_ylabel('True temperature (°C)')
    for ax in axes[1]:ax.set_ylabel('Applied electrical P (kW)')
    fig.suptitle('3 February | controlled reference | exact forecasts and sensor | model error only',fontsize=11)
    save_figure(fig,'F82-03','C0/C3/C8: constant model error and midday drift','w82-model-figure')
    export_table(model_pairs,'model_error_pairs.csv')
    ''')
    response('C',r'''
    State feedback replaces the prediction initial state while retaining the
    fitted coefficients. MPC can revise actions after observing consequences; the
    fixed plan retains its initial actions. C8 agrees with C0 before noon because
    all channels and dynamics agree. Both cases change the physical reference.
    Assess the controller differences separately for cost and each comfort metric.
    ''')
    md('sensor-text',r'''
    **Break after slide 16:** take the additional 15–20 minutes.

    ## 4. Measurement errors, process disturbances and joint effects (slides 17–24)

    **C4 Sensor noise** changes the available reading:

    $$T_k^{\mathrm{meas}}=T_k^{\mathrm{true}}
      +\underbrace{\color{red}{\nu_k}}_{\text{sensor noise}},\qquad
      \nu_k\sim\mathcal N(0,0.2^2)\ {^\circ\mathrm C}.$$

    The true state is unchanged by the measurement. The noisy reading becomes
    the prediction initial condition and can change the next heating action.

    **C5 Sensor bias** adds a constant warm offset to every reading:

    $$T_k^{\mathrm{meas}}=T_k^{\mathrm{true}}
      +\underbrace{\color{red}{\beta_y}}_{\text{sensor bias}},\qquad
      \beta_y=+0.4\ {^\circ\mathrm C}.$$

    Both policies receive the biased initial reading. MPC continues to receive
    that biased reading at every update.

    **C6 Process noise** changes the physical state during the interval:

    $$T_{k+1}^{\mathrm{true}}=(1-b)T_k^{\mathrm{true}}+bT_{o,k}
      +c\,\mathrm{COP}\,P_k^{\mathrm{act}}
      +\underbrace{\color{red}{w_{k+1}}}_{\text{unmodelled physical disturbance}},\qquad
      w_{k+1}\sim\mathcal N(0,0.05^2)\ {^\circ\mathrm C}.$$

    This teaching perturbation represents unmodelled gains or losses over an
    interval. It acts after the action has been selected. In C6 the next reading
    is exact: $T_{k+1}^{\mathrm{meas}}=T_{k+1}^{\mathrm{true}}$.

    For example, suppose the deterministic next temperature is 19.80°C. A
    physical perturbation of +0.05°C gives a true temperature and exact reading
    of 19.85°C. A sensor perturbation of +0.05°C leaves the true temperature at
    19.80°C and reports 19.85°C. The current transition is complete when the new
    reading arrives; MPC can change the following action.

    ### Task D — distinguish the channels (slide 20; 12 min)

    Use 8 minutes to complete the table and interpret the plots, then 4 minutes
    for discussion. Which case has exact observations despite physical noise? What
    can a biased channel tell the controller about its own bias?
    ''')
    task('D',r'''
    | Case | Reading versus true state | Direct effect |
    |---|---|---|
    | C4 Sensor noise | Difference fluctuates as $\nu$ | Imperfect prediction initial state |
    | C5 Sensor bias | Reading is 0.4°C warmer | Systematically warm initial state |
    | C6 Process noise | Reading equals true state | Physical next state changes |

    The controller cannot identify sensor bias from that channel alone. A
    physical disturbance cannot be undone after it has occurred.
    ''',r'''
    | Case | Reading versus true state | Direct effect |
    |---|---|---|
    | C4 Sensor noise | TODO | TODO |
    | C5 Sensor bias | TODO | TODO |
    | C6 Process noise | TODO | TODO |

    TODO: explain why joint effects need not equal summed separate effects.
    ''','markdown')
    code('sensor-figure',r'''
    fig,axes=plt.subplots(2,3,figsize=(12.5,6))
    for col,name in enumerate(['C4 Sensor noise','C5 Sensor bias','C6 Process noise']):
        result=result_at(name);states=result['boundaries'];rows=result['intervals']
        axes[0,col].plot(states.state_time,states.true_temperature_c,color='tab:blue',label='True T')
        axes[0,col].plot(states.state_time,states.measured_temperature_c,color='black',ls=':',label='Available reading')
        comfort(axes[0,col],rows);axes[0,col].set_title(name,fontsize=10)
        axes[1,col].stairs(rows.applied_power_kw,states.state_time,color='tab:blue',label='MPC applied P')
        for ax in axes[:,col]:clock(ax);ax.legend(fontsize=7);ax.set_xlabel('Time')
    axes[0,0].set_ylabel('Indoor temperature (°C)');axes[1,0].set_ylabel('Electrical P (kW)')
    fig.suptitle('3 February | controlled reference | seed 11 | individual error channels',fontsize=11)
    save_figure(fig,'F82-04','C4/C5/C6: observation and physical errors','w82-sensor-figure')
    # Slide-ready individual case panels use exactly the same stored evidence.
    sensor_panel_manifest=[]
    for name in ['C4 Sensor noise','C5 Sensor bias','C6 Process noise']:
        result=result_at(name);states=result['boundaries'];rows=result['intervals']
        fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
        axes[0].plot(states.state_time,states.true_temperature_c,color='tab:blue',label='Simulated true T')
        axes[0].plot(states.state_time,states.measured_temperature_c,color='black',ls=':',label='Available reading')
        comfort(axes[0],rows)
        axes[1].stairs(rows.applied_power_kw,states.state_time,color='tab:blue',label='MPC applied P')
        for ax in axes:clock(ax);ax.set_xlabel('Time');ax.legend(fontsize=8)
        axes[0].set_ylabel('Indoor temperature (°C)');axes[1].set_ylabel('Electrical P (kW)')
        fig.suptitle('3 February | '+name+' | seed 11 | controlled reference',fontsize=11)
        fig.tight_layout(rect=(0,0,1,.94));stem='w8d2_F04_'+name.split()[0]
        if EXPORT_ARTEFACTS:
            for ext in ['png','svg']:fig.savefig(FIGURE_DIR/(stem+'.'+ext),dpi=220)
        plt.close(fig)
        sensor_panel_manifest.append({'id':'F82-04-'+name.split()[0],'case':name,'date':WORKED_DAY,'seed':11,
            'cell_tag':'w82-sensor-figure','png':'figures/week8_day2/'+stem+'.png','svg':'figures/week8_day2/'+stem+'.svg',
            'units':'degrees C;kW','csv_evidence':'worked_runs/2025-02-03_'+name.split()[0]+'_11_MPC_*.csv',
            'source_sha256':json.dumps(sources(ROOT),sort_keys=True)})
    export_table(pd.DataFrame(sensor_panel_manifest),'sensor_panel_manifest.csv')
    ''')
    md('combined-text',r'''
    ### C7: combined errors

    C7 activates the same individual terms together. Red marks the changed
    reference coefficient, physical disturbance, sensor errors and forecast errors:

    $$\begin{aligned}
    T_{k+1}^{\mathrm{true}}&=(1-{\color{red}1.3b})T_k^{\mathrm{true}}
      +{\color{red}1.3b}\,T_{o,k}+c\,\mathrm{COP}\,P_k^{\mathrm{act}}
      +{\color{red}w_{k+1}},\\
    T_k^{\mathrm{meas}}&=T_k^{\mathrm{true}}+{\color{red}\beta_y+\nu_k},\\
    \widehat T_{o,\tau|k}&=T_{o,\tau}+{\color{red}\beta_f+\eta_{k,\tau-k}},\quad \tau>k.
    \end{aligned}$$

    Their magnitudes remain those in C1–C6. The controller retains its original
    coefficients and receives the perturbed forecasts and readings. The
    operating effects interact because each action changes later states and
    decisions; assess C7 using its own simulated trajectories.
    ''')
    code('combined-figure',r'''
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for strategy in ['Fixed schedule','MPC']:
        result=result_at('C7 Combined errors',strategy);states=result['boundaries'];rows=result['intervals']
        axes[0].plot(states.state_time,states.true_temperature_c,color=COLOURS[strategy],label=strategy+' true T')
        if strategy=='MPC':axes[0].plot(states.state_time,states.measured_temperature_c,color='black',ls=':',label='MPC reading')
        axes[1].stairs(rows.applied_power_kw,states.state_time,color=COLOURS[strategy],label=strategy)
    comfort(axes[0],rows)
    for ax in axes:clock(ax);ax.set_xlabel('Time');ax.legend(fontsize=8)
    axes[0].set_ylabel('Indoor temperature (°C)');axes[1].set_ylabel('Electrical P (kW)')
    fig.suptitle('3 February | controlled reference | combined errors | seed 11',fontsize=11)
    save_figure(fig,'F82-05','C7: all declared channels active together','w82-combined-figure')
    ''')
    response('D',r'''
    Each new observation initialises the next prediction. Sensor noise or bias
    can make that initial value inaccurate. In C6 the reading is exact and reports
    the state after a physical disturbance. The controller retains its model
    coefficients throughout. Each isolated experiment changes one source; C7
    changes several sources and therefore changes later actions and states through
    their combined effects.
    ''')
    code('sensor-detail',r"""
    sensor_detail=worked_metrics.loc[worked_metrics['case'].isin(['C4 Sensor noise','C5 Sensor bias','C6 Process noise']),
        ['case','seed','strategy','energy_kwh','bill_gbp','peak_kw','below_degree_hours','above_degree_hours','total_degree_hours','complete']].copy()
    local=sensor_detail.pivot(index=['case','seed'],columns='strategy',values=['bill_gbp','total_degree_hours']).reset_index()
    local.columns=['Case','Seed','Fixed bill (£)','MPC bill (£)','Fixed D (°C h)','MPC D (°C h)']
    local['Case']=local['Case'].str.split().str[0]
    display(sensor_detail.round(4))
    table_plot(local.round(3),'F82-12','3 February | each sensor/process seed retained; D uses simulated true temperature',4.5)
    export_table(sensor_detail,'w8d2_F12_sensor_detail.csv')
    """)
    md('responses-to-errors',r'''
    ### Select a response to the observed limitation (slides 23–24)

    | Limitation | Relevant response and evaluation |
    |---|---|
    | Biased future forecasts | Forecast calibration; validate on new forecast origins and operating outcomes |
    | Model mismatch or changing parameters | Model validation or adaptation; assess prediction and control performance on new conditions |
    | Systematic sensor bias | Calibration or redundant independent sensing; check against a reliable reference |
    | Noisy outputs or hidden states | State estimation/filtering; test assumptions, lag and state accuracy |
    | Material uncertainty in constraints or inputs | Explicit robust/stochastic formulations when justified by uncertainty information; evaluate service and cost |

    The current MPC implementation updates its plan using the available
    information. Calibration, model adaptation and state estimation require the
    additional methods in the table. Robust and stochastic MPC require an
    explicit description of the uncertainty and its role in decisions. Assess true
    comfort from the realised temperatures. Injecting a known error evaluates its
    impact. Detecting and diagnosing an unknown fault requires evidence that
    distinguishes abnormal operation from ordinary noise.
    ''')

    md('metrics-text',r'''
    ## 5. Error effects and controller differences (slides 25–27)

    Retain the 8.1 equipment scope and true endpoint scoring:

    $$E=\Delta t\sum_kP_k^{act},\quad J=\Delta t\sum_kp_kP_k^{act},\quad
    P^{peak}=\max_kP_k^{act},$$

    $$D^-=\Delta t\sum_k\max(L_k^{\mathrm{end}}-T_{k+1}^{true},0),\quad
    D^+=\Delta t\sum_k\max(T_{k+1}^{true}-U_k^{\mathrm{end}},0).$$

    The bill excludes slack penalties. Use true endpoint temperatures for realised
    comfort. Score each applied interval once; predicted slack belongs to the
    separate optimisation problems. The report retains failures, fallback and
    completion. Report full-day metrics for completed runs. Mark metrics from
    incomplete runs as unavailable.

    Compare each case with C0 for the **same strategy** to measure error effects.
    Then compare strategies **within a case** using $\Delta J=J_{MPC}-J_{fixed}$ and
    $\Delta D=D_{MPC}-D_{fixed}$. A lower bill can accompany more discomfort.
    C0 supplies the no-error reference values; its two policies coincide on this day.

    ### Task E — calculate both differences (slide 26; 10 min)

    Merge each run with its policy's C0 metrics, calculate changes from that
    baseline, then form the within-case paired table. Use 7 minutes for analysis
    and 3 for debrief. Retain all seeds and report the minimum and maximum
    observed values as descriptive ranges.
    ''')
    task('E',r'''
    baseline_columns=['strategy','bill_gbp','total_degree_hours']
    baseline=worked_metrics.loc[
        worked_metrics['case']=='C0 No-error reference',baseline_columns]
    baseline=baseline.rename(columns={
        'bill_gbp':'baseline_bill_gbp',
        'total_degree_hours':'baseline_degree_hours'})
    error_effects=worked_metrics.merge(baseline,on='strategy',validate='many_to_one')
    error_effects['bill_change_from_baseline']=error_effects.bill_gbp-error_effects.baseline_bill_gbp
    error_effects['discomfort_change_from_baseline']=error_effects.total_degree_hours-error_effects.baseline_degree_hours
    worked_pairs=pairs(worked_metrics)
    display(error_effects[['case','seed','strategy','bill_change_from_baseline','discomfort_change_from_baseline']].round(4))
    ''',r'''
    # TODO E: merge each run with its strategy's C0 bill/degree-hours.
    # Define error_effects with bill_change_from_baseline and discomfort_change_from_baseline.
    # Define worked_pairs=pairs(worked_metrics).
    raise NotImplementedError('Task E: separate case effects from policy differences.')
    ''')
    code('summary-figures',r'''
    worked_summary=summarise(worked_metrics);names=list(CASES)[1:];x=np.arange(len(names));width=.34
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for strategy,shift in [('Fixed schedule',-width/2),('MPC',width/2)]:
        frame=worked_summary.loc[worked_summary.strategy==strategy].set_index('case').loc[names]
        for ax,col in zip(axes,['bill_gbp','total_degree_hours']):
            means=frame[col+'_mean'].to_numpy();lo=means-frame[col+'_min'].to_numpy();hi=frame[col+'_max'].to_numpy()-means
            ax.bar(x+shift,means,width=width,color=COLOURS[strategy],label=strategy)
            repeated=frame.valid.to_numpy()>1
            ax.errorbar((x+shift)[repeated],means[repeated],yerr=np.vstack([lo[repeated],hi[repeated]]),fmt='none',capsize=3,color='black')
    reference_bill=worked_summary.loc[worked_summary['case']=='C0 No-error reference','bill_gbp_mean'].mean()
    axes[0].axhline(reference_bill,color='.45',ls='--',lw=1,label='C0 no-error bill')
    for ax in axes:ax.set_xticks(x,[name.split()[0] for name in names]);ax.set_xlabel('Error case');ax.legend(fontsize=8);ax.grid(axis='x',visible=False)
    axes[0].set_ylabel('HP bill (£)');axes[1].set_ylabel('True discomfort (°C h)')
    fig.suptitle('3 February | bars = mean; whiskers = observed min–max | stochastic n=3, deterministic n=1',fontsize=10)
    save_figure(fig,'F82-06','C1–C8 grouped bars; observed seed ranges; C0 shown as a bill reference','w82-summary-figures')
    rows=[]
    for name in CASES:
        row={'Case':name,'n':3 if CASES[name].stochastic else 1}
        for strategy,label in [('Fixed schedule','Fixed'),('MPC','MPC')]:
            entry=worked_summary.set_index(['case','strategy']).loc[(name,strategy)]
            for col,title in [('bill_gbp','bill (£)'),('total_degree_hours','D (°C h)')]:
                row[label+' '+title]=f"{entry[col+'_mean']:.3f} [{entry[col+'_min']:.3f}, {entry[col+'_max']:.3f}]"
        rows.append(row)
    operating_table=pd.DataFrame(rows);display(operating_table)
    table_plot(operating_table,'F82-09','Worked mean [observed min,max] | descriptive ranges across supplied seeds',4.5)
    export_table(worked_summary,'worked_summary.csv');export_table(error_effects,'error_effects_from_baseline.csv')
    export_table(worked_pairs,'worked_paired_differences.csv');export_table(operating_table,'worked_operating_table.csv')
    ''')
    response('E',r'''
    Case-minus-baseline compares the effect of an error for a given policy;
    MPC-minus-fixed compares policies exposed to the same error. Cost, demand and
    discomfort can move differently. The error settings have different magnitudes
    and units. Interpret each case under its supplied settings when comparing the
    outcomes. Three seeds support descriptive ranges only. Deterministic C0 is
    independent of the unused noise arrays. One C0 run per strategy therefore
    supplies the common baseline.
    ''')
    md('final-text',r'''
    ## 6. Freeze settings and evaluate new dates (slides 28–32)

    Freeze all coefficients, error settings, horizon, penalty and update rules
    before evaluating 9–11 February. Use C1/C2/C3/C7 and fixed/MPC. C1/C7 have
    three seeds; deterministic C2/C3 have one. This gives 48 runs and 24 matched
    pairs, with a physical reset for each date. The same seed innovations repeat
    across dates, so outcomes on those dates share the same random inputs.
    C4/C5/C6 are absent from the final set. C1/C7 contribute more seed rows than
    C2/C3, so pooled counts give them more weight. Counts describe the evaluated
    cases and seeds. Retain every pair and failure.

    The separate transfer example returns to the unchanged richer two-state
    reference from 8.1, original supplied forecasts, 9 February and seed 90. Its
    errors are combined/unspecified, distinct from controlled C7. Compare policies
    within each reference model when interpreting costs.
    ''')
    code('final-batch',r'''
    frozen_protocol=freeze_protocol(ROOT,DEVELOPMENT_DATES,FINAL_DATES)
    export_json(frozen_protocol,'frozen_protocol.json');verify_protocol(ROOT,frozen_protocol)
    started=time.perf_counter()
    final_metrics,final_runs=batch(operation,model,frozen_protocol['final_cases'],FINAL_DATES,SEEDS)
    final_runtime_s=time.perf_counter()-started;final_pairs=pairs(final_metrics)
    export_table(final_metrics,'final_metrics.csv');export_table(final_pairs,'final_paired_differences.csv')
    export_table(final_metrics.loc[~final_metrics.complete],'final_failed_runs.csv')
    ''')
    md('task-f-text',r'''
    ### Task F — state what the evidence supports (slide 29; 10 min)

    Report valid/invalid counts and mean cost/comfort differences by case before
    pooled counts. Contrast these results with worked C5 sensor bias before
    recommending a controller. State a relevant response and one limitation. Use 7
    minutes for analysis and 3 for debrief. Base the recommendation on the
    computed cost and comfort results.
    ''')
    task('F',r'''
    final_case_evidence=case_evidence(final_pairs)
    display(final_case_evidence.round(4))
    valid_final=final_pairs.loc[final_pairs.valid_pair]
    evidence={'valid_pairs':len(valid_final),'invalid_pairs':int((~final_pairs.valid_pair).sum()),
        'mpc_lower_bill':int((valid_final.delta_bill_gbp<0).sum()),
        'mpc_lower_discomfort':int((valid_final.delta_total_degree_hours<0).sum()),
        'mean_bill_difference_gbp':float(valid_final.delta_bill_gbp.mean()),
        'mean_discomfort_difference_degree_hours':float(valid_final.delta_total_degree_hours.mean())}
    display(pd.Series(evidence))
    ''',r'''
    # TODO F: define final_case_evidence=case_evidence(final_pairs); display it first.
    # Then define valid_final and evidence with valid/invalid pair counts,
    # mpc_lower_bill, mpc_lower_discomfort, mean_bill_difference_gbp,
    # and mean_discomfort_difference_degree_hours.
    raise NotImplementedError('Task F: summarise all held-out paired outcomes.')
    ''')
    code('final-figure',r'''
    rows=[]
    for name in frozen_protocol['final_cases']:
        group=valid_final.loc[valid_final['case']==name]
        row={'case':name,'valid_pairs':len(group)}
        for metric in ['delta_bill_gbp','delta_total_degree_hours']:
            row[metric+'_mean']=group[metric].mean();row[metric+'_min']=group[metric].min();row[metric+'_max']=group[metric].max()
        rows.append(row)
    final_bar_summary=pd.DataFrame(rows);x=np.arange(len(final_bar_summary))
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for ax,metric,label in zip(axes,['delta_bill_gbp','delta_total_degree_hours'],['Bill: MPC − fixed (£)','True discomfort: MPC − fixed (°C h)']):
        means=final_bar_summary[metric+'_mean'].to_numpy()
        lo=means-final_bar_summary[metric+'_min'].to_numpy();hi=final_bar_summary[metric+'_max'].to_numpy()-means
        ax.bar(x,means,width=.55,color='tab:blue')
        ax.errorbar(x,means,yerr=np.vstack([lo,hi]),fmt='none',color='black',capsize=4)
        ax.axhline(0,color='.4',lw=1);ax.set_ylabel(label);ax.grid(axis='x',visible=False)
        ax.set_xticks(x,[f'{name.split()[0]}\n(n={n})' for name,n in zip(final_bar_summary['case'],final_bar_summary.valid_pairs)])
    fig.suptitle('Held-out dates | bars = mean paired difference; whiskers = observed min–max | descriptive ranges',fontsize=10)
    save_figure(fig,'F82-07','Held-out C1/C2/C3/C7 paired differences; repeated seed streams across dates','w82-final-figure')
    export_json(evidence,'final_evidence.json')
    export_table(final_case_evidence,'final_case_evidence.csv')
    export_table(final_bar_summary,'held_out_bar_summary.csv')
    ''')
    code('transfer',r'''
    transfer_runs={}
    for strategy in ['Fixed schedule','MPC']:
        transfer_runs[strategy]=run_experiment(
            strategy,operation,supplied_forecasts,model,
            day=FINAL_DATES[0],M=M,N=N,seed=90,gamma=gamma)
    transfer_metrics=pd.DataFrame([realised_metrics(result) for result in transfer_runs.values()])
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for strategy,result in transfer_runs.items():
        states=result['boundaries'];rows=result['intervals']
        axes[0].plot(states.state_time,states.true_temperature_c,color=COLOURS[strategy],label=strategy)
        axes[1].stairs(rows.applied_power_kw,states.state_time,color=COLOURS[strategy],label=strategy)
    comfort(axes[0],rows)
    for ax in axes:clock(ax,FINAL_DATES[0]);ax.set_xlabel('Time');ax.legend(fontsize=8)
    axes[0].set_ylabel('True temperature (°C)');axes[1].set_ylabel('Electrical P (kW)')
    fig.suptitle('9 February | richer two-state reference | original forecast CSV | seed 90',fontsize=11)
    save_figure(fig,'F82-08','Separate richer-reference transfer; combined/unspecified errors','w82-transfer')
    display(transfer_metrics.round(4));export_table(transfer_metrics,'richer_reference_transfer_metrics.csv')
    ''')
    response('F',r'''
    Report each case before pooling. Compare the final results with C5, where MPC
    has more true discomfort than the fixed schedule under persistently warm
    readings. The final set excludes this sensor-only case. Use the computed
    counts, individual pairs and cases together. They establish what happened on
    three new dates under declared error settings and a small seed set. Broader
    policy comparisons, error probabilities and comfort guarantees require further
    evidence. The richer-reference transfer connects to 8.1 with several error
    sources active together.
    ''')
    md('learning-check',r'**Learning check (slide 32):** explain one error mechanism, one feedback limitation and one justified next response.')
    cells.extend(extension_cells(solution))
    code('exports',r'''
    export_runs(worked_runs,'worked_runs');export_runs(final_runs,'final_runs')
    export_runs({(FINAL_DATES[0],'Richer',90,s):v for s,v in transfer_runs.items()},'transfer_runs')
    if EXPORT_ARTEFACTS:pd.DataFrame(figure_manifest).to_csv(ROOT/'teaching/week8_day2_figure_manifest.csv',index=False)
    export_json({'source_sha256':sources(ROOT),'N':N,'M':M,'dt_hours':dt,'gamma':gamma,
        'seeds':SEEDS,'worked_day':WORKED_DAY,'final_dates':FINAL_DATES,
        'worked_run_count':len(worked_metrics),'final_run_count':len(final_metrics),
        'stochastic_extension_run_count':len(stochastic_metrics),'stochastic_extension_scenarios':S,
        'stochastic_extension_horizon':HORIZON,'stochastic_extension_realisations':N_REALISATIONS,
        'stochastic_extension_protocol':stochastic_protocol,
        'worked_runtime_s':worked_runtime_s,'final_runtime_s':final_runtime_s,'cached_results':False,
        'solver':SOLVER,'solver_options':solver_options,'python':platform.python_version(),
        'reference_protocol':'No-error and perturbed one-state experiments; richer two-state transfer is separate.'},'run_metadata.json')
    ''')
    md('references',r'''
    ## References and source credits (slide 33)

    - Tutorial 8.1: MPC formulation, visible first-action loop, reference-model appendix and equipment metrics.
    - Powell, W. B. (2019), [*A unified framework for stochastic optimization*](https://warrenpowell.org/assets/papers/2019/02/Powell-A-Unified-Framework-for-Stochastic-Optimization-EJOR.pdf), European Journal of Operational Research 275, 795–821, Sections 4 and 9.4: information timing and operating/lookahead models. The tutorial retains the course hat and bar conventions.
    - Tutorials 1.2, 4.1b and 4.2: forecast origin, issue/valid time, lead time and actual-minus-forecast error.
    - Mesbah, A. (2016), [*Stochastic Model Predictive Control: An Overview and Perspectives for Future Research*](https://escholarship.org/content/qt1wt3d4vr/qt1wt3d4vr.pdf), IEEE Control Systems Magazine 36(6), 30–44: scenario approximations and the role of future information in control decisions.
    - [Reference data README](data/reference_building/README.md): synthetic teaching data and provenance limits.
    - Drgoňa et al. (2020), *All you need to know about model predictive control for buildings*, Annual Reviews in Control 50, 190–232. DOI: 10.1016/j.arcontrol.2020.09.001. The perturbation values here are illustrative course settings.
    ''')
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','language':'python','display_name':'Python 3'},'language_info':{'name':'python'}})

def write_metadata():
    with (ROOT/'teaching/week8_day2_task_manifest.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['task_id','slide','minutes_including_debrief','student_work_minutes','debrief_minutes']);w.writeheader()
        for letter,slide,minutes in TASKS:
            d=4 if letter in 'ABD' else 3
            w.writerow(dict(task_id=letter,slide=slide,minutes_including_debrief=minutes,student_work_minutes=minutes-d,debrief_minutes=d))
        for letter,minutes,debrief in EXTENSION_TASKS:
            w.writerow(dict(task_id=letter,slide='Extension',minutes_including_debrief=minutes,student_work_minutes=minutes-debrief,debrief_minutes=debrief))
    with (ROOT/'teaching/week8_day2_schedule.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['slide','block','core_minutes','task','break_after_minutes','guided_minutes']);w.writeheader()
        for slide,minutes in enumerate(SLIDE_MINUTES,1):w.writerow(dict(slide=slide,block=(slide-1)//8+1,core_minutes=minutes,task=next((x[0] for x in TASKS if x[1]==slide),''),break_after_minutes='15–20 additional' if slide==16 else '',guided_minutes={7:4,8:4,9:2,10:2,11:2,13:4,16:2,17:2,18:2,19:2,21:3,30:3}.get(slide,0)))

if __name__=='__main__':
    for solution in [False,True]:nbf.write(make_notebook(solution),ROOT/(STEM+('_solution' if solution else '')+'.ipynb'))
    write_metadata();print('Built Tutorial 8.2: isolated errors, then joint cases and richer-reference transfer.')
