"""Build the concept-led Tutorial 8.1 pair; preserve the reviewed controller."""
from pathlib import Path
import csv
import textwrap
import nbformat as nbf
ROOT = Path(__file__).resolve().parents[1]
STEM = 'tutorial_8_1_model_predictive_control'
TASKS = [('A', 8, 8), ('B', 16, 15), ('C', 18, 8), ('D', 24, 22), ('E', 29, 10), ('F', 33, 8)]
SLIDE_MINUTES = [2, 3, 5, 6, 6, 6, 6, 8, 8, 3, 4, 3, 4, 5, 3, 15, 5, 8, 3, 3, 3, 4, 7, 22, 4, 2, 2, 3, 10, 4, 3, 3, 8, 5, 12, 2]
FIGURES = [('F81-01', 21, 'w8d1_F01_first_plan', 'First forecast-based plan: predicted temperature, electrical power, prediction initial condition and first applied interval.', 'B'), ('F81-02', 25, 'w8d1_F02_successive_plans', 'Initial 00:00 prediction and plan versus realised MPC operation and stored one-step predictions; full plans remain exported.', 'D'), ('F81-03', 29, 'w8d1_F03_fixed_vs_mpc', 'Fixed first-forecast plan versus MPC under common reference conditions; bill and both comfort deviations.', 'E'), ('F81-04', 33, 'w8d1_F04_thermostat_context', 'Add the Week 7 midpoint thermostat: heat-pump electricity, bill, peak and true endpoint temperatures/comfort.', 'F'), ('F81-03-table', 29, 'w8d1_F03_fixed_vs_mpc_metrics', 'Separate fixed/MPC bill and original-bound comfort table for the nominal reference.', 'E'), ('F81-06', 34, 'w8d1_F06_daily_cost', 'Daily electricity bill: interval costs, cumulative totals and averages.', 'Guided')]

def make_notebook(solution=False):
    cells = []

    def md(key, source, tags=None):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(), id='w81-' + key, metadata={'tags': tags or []}))

    def code(key, source, tags=None):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(), id='w81-' + key, metadata={'tags': tags or ['supplied']}))

    def task(letter, answer, scaffold, kind='code'):
        (code if kind == 'code' else md)('task-' + letter.lower(), answer if solution else scaffold, ['task', letter])

    def response(letter, prompt, answer):
        md('response-' + letter.lower(), '**Interpretation ' + letter + '.** ' + prompt + r"""

""" + (answer if solution else '*Write your interpretation here.*'))
    md('purpose', r'''
    # Tutorial 8.1: Model predictive control for buildings

    In Week 7 we optimised a heating schedule over 24 hours. Following the whole
    schedule gives a fixed *open-loop* plan. Here we turn it into model predictive
    control (MPC): measure the indoor temperature, optimise a heating plan, apply
    its first action, then repeat with the next measurement.

    The core tutorial develops this loop step by step. Tasks A–D cover the inputs,
    formulation and implementation. Tasks E–F compare realised cost and comfort
    with a fixed plan and the Week 7 thermostat. We then calculate daily totals
    and averages and discuss the information each policy requires.

    After completing the core, choose any optional activity: temperature tracking,
    heating with uncertain PV and a battery, or a horizon that shrinks to midnight. Each activity includes the setup needed to run
    independently
    of the other optional activities. Appendix A gives the reference model and a
    thermal-network schematic.

    Run the notebook from the repository root. Complete the core cells in order;
    the student copy leaves Tasks A–F and interpretation answers for you to
    finish.
    ''')

    code('setup', r"""from pathlib import Path
import hashlib
import json
import platform
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scripts.week8_day1_visuals import concept
import cvxpy as cp
from IPython.display import display
from course_utils.week8_control import (
    announced_schedule, forecast_rows, schedule_horizon, validate_weather_vector,
    SimulationSession, solve_update, first_action, FallbackPolicy, choose_command,
    rejected_plan, solver_configuration, realised_metrics)

ROOT=Path.cwd()
DATA_DIR=ROOT/'data/reference_building'
if not (DATA_DIR/'operation_conditions.csv').is_file():raise FileNotFoundError('Run from the repository root.')
N=48;M=48;SEED=88;DAY='2025-02-03';gamma=25.;comfort_margin_c=0.
SOLVER,_solver_options=solver_configuration()
EXPORT_ARTEFACTS=True
FIGURE_DIR=ROOT/'figures/week8_day1';OUTPUT_DIR=ROOT/'teaching/week8_day1_outputs'
plt.rcParams.update({'font.size':12,'figure.dpi':110,'axes.grid':True,'grid.alpha':.2,
                     'svg.fonttype':'none','savefig.facecolor':'white'})
COLOURS={'Thermostat':'tab:red','Fixed schedule':'tab:green','MPC':'tab:blue'}
print('Solver:',SOLVER,'| Reporting day:',DAY,'| Forecast horizon:',N,'half-hours')""")

    md('signals', r'''
    ## 1. From a schedule to feedback control (slides 3–9)

    A building automation system (BAS) connects sensors, local controllers and
    actuators. MPC can provide supervisory optimisation within this system, for
    example by sending temperature setpoints to HVAC equipment. In this teaching
    case, MPC sets electrical heating power directly. This is one form of
    automated optimisation within the EMIS context introduced in Week 3.

    The **state** describes the system's evolving condition and carries
    information from one interval to the next. Our prediction model uses indoor
    temperature $T$ as its state. The **control**, or decision,
    $P$ is electrical heating power in kW. The **measurement**
    $T^{\mathrm{meas}}$ is a sensor reading that may differ from the underlying
    temperature. **Disturbances**, including outdoor temperature and heat gains,
    are external inputs supplied to the heating controller. We assume they evolve
    independently of its heating actions. Price enters the objective, and the
    imposed comfort bounds constrain predicted temperature. The $1R1C$
    coefficients $b,c$ remain fixed throughout the simulation;  they
    can be estimated from observations using the Week 6 methods before
    implementing the controller.

    Two prediction tasks are required. First, a weather forecast describes future
    outdoor conditions. Second, the thermal model predicts indoor temperatures
    conditional on that weather forecast, the current measurement and a candidate
    heating sequence. Changing the heating actions changes the predicted indoor
    temperature. The supplied weather forecast stays fixed within that
    optimisation.

    Electricity price is an exogenous economic input in the objective. The
    time-of-use tariff is known over the horizon, and comfort bounds specify the
    operating requirements. Future outdoor temperature is supplied as an imperfect
    weather forecast from the CSV; only the current outdoor nowcast is assumed
    exact.
    ''')

    md('case-overview', r'''
    ### The heating case study

    We operate a heat pump over 3 February 2025 and compare three policies: *(i)*
    a fixed plan optimised once at midnight, *(ii)* MPC that replans every 30
    minutes, and *(iii)* the Week 7 thermostat. The aim is to compare electrical
    energy, bill, peak demand and indoor comfort under the same operating
    conditions.

    The prediction and reference models have separate roles:

    | Component | Inputs and role | Information available to the controller |
    |---|---|---|
    | One-state $1R1C$ prediction model | Current measured indoor temperature, candidate heating and forecast outdoor temperature → predicted indoor temperatures | Its supplied coefficients and all planning inputs |
    | Richer two-state **reference** model | Applied heating, realised outdoor weather and heat gains → simulated air/fabric temperatures and a noisy sensor reading | The returned indoor measurement; underlying air and fabric temperatures are used only to check performance |

    The reference model is provided in `course_utils/reference_building.py`, with
    loading and simulation support supplied. Its additional state, gains and
    temperature-dependent COP create model mismatch with the controller's simpler
    model. The weather forecast CSV is a separate information source and  can
    differ from the realised outdoor weather driving the reference. Thus this
    example includes model mismatch, forecast error and sensor/process noise from
    the outset. In Tutorial 8.2, we will examine the effect of each uncertainty
    source separately.

    The midnight optimisation predicts a full day's indoor temperatures, but only
    the reference model determines the realised trajectory. MPC receives a new
    measurement before revising its next plan. The fixed open-loop policy
    continues its original sequence. We will plot the initial prediction beside
    both realised trajectories to make this distinction visible.
    ''')

    code('plot-helpers', 'FIGURE_SPECS=' + repr(FIGURES) + r"""
""" + r"""figure_manifest=[]
source_paths=['data/reference_building/operation_conditions.csv','data/reference_building/operation_forecasts.csv',
    'data/reference_building/verified_controller_model.json','course_utils/reference_building.py',
    'scripts/week8_review_diagnostics.py','scripts/week8_day1_transfer.py','scripts/week8_day1_optional.py','course_utils/week8_control.py','scripts/build_week8_day1.py','scripts/week8_day1_visuals.py']

def export_table(frame,name):
    if EXPORT_ARTEFACTS:
        OUTPUT_DIR.mkdir(parents=True,exist_ok=True);frame.to_csv(OUTPUT_DIR/name,index=False)

def save_figure(fig,figure_id,case=DAY):
    spec=next(row for row in FIGURE_SPECS if row[0]==figure_id)
    label = 'Nominal reference' if case==DAY else case
    fig.suptitle('3 February 2025 | '+label+' | supplied forecasts | N=48, γ=25',fontsize=10)
    fig.tight_layout(rect=(0,0,1,.94))
    if EXPORT_ARTEFACTS:
        FIGURE_DIR.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg']:fig.savefig(FIGURE_DIR/f'{spec[2]}.{ext}',dpi=220)
        figure_manifest.append({'id':figure_id,'notebook':'tutorial_8_1_model_predictive_control_solution.ipynb','cell_tag':{'F81-06':'w81-daily-cost','F81-01':'w81-first-plan-figure','F81-02':'w81-successive-plans','F81-03':'w81-fixed-comparison','F81-04':'w81-thermostat-comparison','F81-03-table':'w81-fixed-comparison-table','F81-05':'w81-heat-loss-runs','F81-05-table':'w81-heat-loss-table'}[figure_id],'reference_model':'Two-state air/fabric synthetic reference','completion_status':'complete','evidence_stem':{'F81-06':'w8d1_F06_daily_cost','F81-01':'first_plan','F81-02':'successive_plans','F81-03':'fixed_vs_mpc','F81-04':'three_policy_metrics','F81-03-table':'fixed_vs_mpc','F81-05':'heat_loss_sensitivity','F81-05-table':'heat_loss_sensitivity'}[figure_id],'slide':spec[1],'task':spec[4],'caption':spec[3],
            'png':f'figures/week8_day1/{spec[2]}.png','svg':f'figures/week8_day1/{spec[2]}.svg',
            'placement':'12.5 x 3.5 inch frame; preserve aspect ratio','case':case,
            'period':str(state_time[0])+' to '+str(state_time[-1]),'N':N,'M':M,'dt_hours':dt,
            'seed':SEED,'gamma_gbp_per_degree_hour':gamma,'comfort_margin_c':comfort_margin_c,'reference_resistance_multiplier':.8 if figure_id.startswith('F81-05') else 1.,
            'units':'kW;kWh;GBP;degrees C;degrees C h;s','assumptions':
                'Current nowcast; future issue-specific forecasts; noisy measured initial condition; lower-soft upper-hard predictions; independent plant; original endpoint comfort',
            'source_files':';'.join(source_paths),
            'source_sha256':';'.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths)})
    plt.show();plt.close(fig)

def clock_axis(ax,end=None):
    ax.set_xlim(state_time[0],end if end is not None else state_time[-1])
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,4)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

def comfort_lines(ax,starts,lower,upper):
    ends=pd.DatetimeIndex(starts)+pd.Timedelta(hours=dt)
    ax.plot(ends,lower,'.--',color='0.35',markersize=3,label='Original endpoint bounds')
    ax.plot(ends,upper,'.--',color='0.35',markersize=3)

def draw_table(ax,frame,fontsize=11,widths=None,scale=1.7):
    shown=frame.copy()
    for col in shown.select_dtypes(include='floating'):
        shown[col]=shown[col].map(lambda x:'n/a' if pd.isna(x) else f'{0. if round(x,3)==0 else x:.3f}')
    ax.axis('off');table=ax.table(cellText=shown.values,colLabels=shown.columns,loc='center',
        cellLoc='center',colWidths=widths)
    table.auto_set_font_size(False);table.set_fontsize(fontsize);table.scale(1,scale)
    for (row,col),cell in table.get_celld().items():
        cell.set_edgecolor('0.8')
        if row==0:cell.set_facecolor('0.92')
    return table

def export_run(result,prefix):
    export_table(result['intervals'],prefix+'_intervals.csv')
    export_table(result['boundaries'],prefix+'_boundaries.csv')
    if len(result['forecast_records']):export_table(result['forecast_records'],prefix+'_forecast_inputs.csv')
    for plan in result['plans']:
        tag=pd.Timestamp(plan['origin']).strftime('%H%M')
        export_table(pd.DataFrame({'interval_start':plan['valid_time'],'power_kw':plan['power_kw'],
            'outdoor_forecast_c':plan['outdoor'],'price':plan['price'],'lower':plan['lower'],'upper':plan['upper'],
            'predicted_lower_slack_c':plan['slack_c']}),prefix+'_plan_'+tag+'_intervals.csv')
        export_table(pd.DataFrame({'state_time':plan['state_time'],'predicted_temperature_c':plan['temperature_c']}),
            prefix+'_plan_'+tag+'_states.csv')

concept_manifest=[]
def show_concept(figure_id,slides):
    fig=concept(figure_id)
    if EXPORT_ARTEFACTS:
        FIGURE_DIR.mkdir(parents=True,exist_ok=True)
        for ext in ['png','svg']:fig.savefig(FIGURE_DIR/f'{figure_id}.{ext}',dpi=220)
        concept_manifest.append({'id':figure_id,'slides':slides,'kind':'schematic; not numerical evidence',
            'source':'scripts/week8_day1_visuals.py','svg':f'figures/week8_day1/{figure_id}.svg',
            'source_sha256':hashlib.sha256((ROOT/'scripts/week8_day1_visuals.py').read_bytes()).hexdigest()})
    plt.show();plt.close(fig)
""")

    code('loop-diagram', "show_concept('C81-01','3,4,7,9')")

    md('task-a-text', r'''
    ### Task A — identify the signals (slide 8; 8 min)

    Work in pairs for 5 minutes, then compare answers for 3 minutes. Complete the
    table with each signal's role and the information available to the controller.
    Distinguish current measurements, future forecasts and simulated states
    available only to the evaluator.
    ''')

    task('A', r"""| Signal | Role | Controller access |
|---|---|---|
| Indoor temperature | State $T$; observed output $T^{\mathrm{meas}}$ | Current sensor reading; true state is used only to check performance |
| Outdoor temperature | Exogenous input $T_o$ | Current nowcast and forecasts of future values |
| Electrical heating power | Chosen control $P$ | Proposed commands; actual power logged after application |
| Electricity price | Economic input $p$ | Provided schedule |
| Comfort limits | Bounds $L,U$ | Provided schedule |
| Fabric temperature | Hidden reference state | Unavailable to this controller |""", r"""| Signal | Role | Controller access |
|---|---|---|
| Indoor temperature | TODO | TODO |
| Outdoor temperature | TODO | TODO |
| Electrical heating power | TODO | TODO |
| Electricity price | TODO | TODO |
| Comfort limits | TODO | TODO |
| Fabric temperature | TODO | TODO |""", 'markdown')

    code('prediction-diagram', r"""# Supplied debrief illustration: two distinct prediction tasks.
show_concept('C81-02','5,6,8,20')""")

    md('horizons', r'''
    ## 2. Horizons and repeated optimisation (slides 10–18)

    Each action is held for $\Delta t=0.5$ h. A prediction horizon of
    $N=48$ therefore covers 24 hours, with 48 controls and 49
    temperature boundaries. Here all $N_c=N$ future moves are
    independent decision variables. Applying only the first move leaves the
    remaining moves as provisional decisions; $N_c=1$ would instead
    restrict the control horizon.

    In the standard **rolling** or **receding-horizon** implementation used here,
    the origin advances by one interval while the prediction duration stays fixed.
    For example, at 00:00 we plan to 24:00; at 00:30 we plan to 00:30 on the next
    day. The planning window advances while retaining its duration. A **shrinking
    horizon** instead keeps the midnight endpoint: at 00:30 only 23.5 hours would
    remain. This distinction follows the standard [MPC
    description](https://www.do-mpc.com/en/latest/theory_mpc.html).

    In Week 7, we solved one problem over a fixed operating period. Following its
    entire action sequence without revision gives the fixed open-loop benchmark
    used here. MPC repeatedly solves a related problem with a new starting state
    and later endpoint. These are operating horizons, distinct from forecasting
    training windows. The schematic labels the comfort band and first applied
    interval separately from the forecast information.

    Use the following six steps throughout this tutorial. They follow the
    state initialisation, disturbance prediction, optimisation and first-action
    application described by Drgoňa et al. (2020):

    1. **Read measurement:** read the current indoor measurement.
    2. **Gather inputs:** obtain the outdoor nowcast, future forecasts and provided
       price/comfort schedules over the next $N$ intervals.
    3. **Optimise and check:** initialise the prediction at that measurement,
       solve for $N$ controls and $N+1$ temperatures, and validate the solution.
    4. **Select action:** take only $P^*_{k|k}$ from an accepted plan.
    5. **Apply and observe:** hold the action for one interval; the building
       evolves and returns the next indoor measurement.
    6. **Repeat:** start the next iteration from that measurement, retaining
       the same $N$ in the core example.

    In the code, `available()` reads the stored observation. `advance()` performs
    the physical transition in step 5 and stores the new reading. Step 6 is the
    next loop iteration. Each iteration therefore has one physical transition.

    The thermostat also uses feedback. MPC additionally predicts consequences and
    optimises a provisional sequence subject to an objective and constraints.
    ''')

    code('horizon-diagram', r"""show_concept('C81-03','10,11,12,13,18')
# The six-step algorithm is given as the Markdown list above.""")

    md('model-and-data', r'''
    ### Supplied model and operating inputs

    Retain $a=1-b$ from Tutorials 6.1–6.2. The planning time is $k$ and
    the absolute target time is $\tau=k+j$, where $j$ counts intervals ahead.
    The index after the bar records when the plan was made:

    $$\widehat T_{\tau+1|k}=(1-b)\widehat T_{\tau|k}+b\widehat T_{o,\tau|k}+c\,\mathrm{COP}\,P_{\tau|k}.$$

    The coefficient $c$ is discrete and already includes the
    sampling effect; multiplying the thermal input by another $\Delta t$
    would apply the time step twice. $P$ is electrical power, while
    $\mathrm{COP}\,P$ is useful heat rate.

    We use the following sources, with loading and time checks supplied:

    | Source | Role |
    |---|---|
    | `verified_controller_model.json` | Common $b,c$, COP, capacity and time step; fitted over the full identification period |
    | `operation_conditions.csv` | Provided tariff/comfort; current outdoor nowcast; realised weather/gains supplied separately to the reference model |
    | `operation_forecasts.csv` | Pre-generated synthetic outdoor forecasts available at each origin; no forecasting model is trained here |
    | `course_utils/reference_building.py` | Richer synthetic air/fabric reference, introduced below |

    The main day is 3 February 2025, seed 88, with $M=48$ reporting
    intervals. Every MPC update retains $N=48$, including the final
    update at 23:30. Its last predicted boundary is 23:30 on the following day.
    The supplied checks verify forecast and schedule coverage beyond the reporting
    day.

    A hat $\widehat \cdot$ indicates an estimated or predicted quantity, such as
    a fitted model's indoor temperature prediction or a future weather forecast.
    Measurement accuracy is specified separately by the sensor assumptions.
    $\widehat T_{k|k}$ is the first entry of the predicted trajectory and is
    constrained to equal the available measurement $T_k^{\mathrm{meas}}$. Measurement
    error is represented separately by the sensor model. Current outdoor
    temperature at $j=0$ is the exact-nowcast assumption, even though
    it is stored in the same outdoor-input vector as future forecasts.

    The supplied controller values are:

    | Parameter | Value | Unit / meaning |
    |---|---:|---|
    | $b$ | 0.0234549361 | Loss coefficient per 30-minute step |
    | $a=1-b$ | 0.9765450639 | State persistence |
    | $c$ | 0.1261192467 | °C/kW per step; discrete heating coefficient before COP |
    | $\mathrm{COP}$ | 3.0 | Constant prediction-model heat/electricity ratio |
    | $P_{\max}$ | 5 | Electrical kW |
    | $\Delta t$ | 0.5 | h |
    | $N,M$ | 48, 48 | Prediction/reporting intervals |
    | $\gamma$ | 25 | £/(°C h) |

    The equivalent fitted effective values are $R=5.37709$ K/kW and
    $C=3.96450$ kWh/K. Appendix A describes the full reference model,
    including numerical parameters and noise assumptions.
    ''')

    code('inputs', r"""operation=pd.read_csv(DATA_DIR/'operation_conditions.csv',index_col='timestamp',parse_dates=True)
forecasts=pd.read_csv(DATA_DIR/'operation_forecasts.csv',parse_dates=['issue_time','valid_time'])
model=json.loads((DATA_DIR/'verified_controller_model.json').read_text(encoding='utf8'))
dt=float(model['sampling_interval_hours']);b=float(model['loss_coefficient_per_step'])
c=float(model['heat_coefficient_c_per_kw_step']);COP=float(model['nominal_cop'])
P_MAX=float(model['heat_pump_max_electric_kw']);a=1-b
if dt!=.5 or not operation.index.is_unique or not operation.index.is_monotonic_increasing:
    raise ValueError('Inspect the half-hour data index and model time step.')
announced=announced_schedule(operation)  # tariff and comfort only; no future realised weather
reporting_times=pd.date_range(DAY,periods=M,freq='30min')
state_time=pd.date_range(DAY,periods=M+1,freq='30min')
for origin in reporting_times:
    forecast_rows(forecasts,origin,N,dt)
    schedule_horizon(announced,origin,N,dt)
demo=SimulationSession(operation,model,DAY,M=1,seed=SEED)
first_view=demo.available()
first_origin=first_view['origin']
first_future=forecast_rows(forecasts,first_origin,N,dt)
first_schedule=schedule_horizon(announced,first_origin,N,dt)
display(pd.Series({'first_origin':first_origin,'available_indoor_c':first_view['measurement_c'],
    'outdoor_nowcast_c':first_view['nowcast_c'],'reporting_intervals':M,'prediction_intervals':N,
    'last_origin':reporting_times[-1],'last_plan_final_boundary':reporting_times[-1]+pd.Timedelta(hours=N*dt)}))
outdoor_vector=np.r_[first_view['nowcast_c'],first_future.outdoor_temperature_forecast_c.to_numpy()]
forecast_alignment=validate_weather_vector(outdoor_vector,first_origin,first_view['nowcast_c'],first_future,N,dt)
display(forecast_alignment.iloc[[0,1,47]])""")

    md('forecast-timeline', r'''
    A **forecast origin** is when the forecast becomes available. A **lead** is
    how far ahead it refers to, and its **valid time** is the corresponding time:
    valid time = origin + lead × 30 minutes. At 00:00, use the current outdoor
    nowcast for the first interval and the 00:00 forecast for 00:30, 01:00, …,
    23:30. At 00:30, use the new 00:30 forecast and a new current nowcast. The
    table above shows this timeline in file columns; the controller uses the
    forecast available at that origin.

    This matches Tutorial 1.2's **forecast origin** and **lead time**, and
    Tutorial 4.1b's **issue time** and **valid time**. Origin and issue time
    coincide in this supplied table. Lead time is $j\Delta t$; the target
    timestamp is the valid time. A prediction horizon of 24 hours contains 48
    half-hour intervals. As in the earlier forecasting tutorials, observations
    become available only when their timestamps have been reached. Use the issue
    time to determine which forecast was available for each decision. Forecast
    error uses actual minus forecast, consistent with Week 4.2.
    ''')

    md('formulation-text', r'''
    ### The optimisation at each interval (slides 15–17)

    Use integer indices throughout. The current interval is $k$ and
    the number of intervals ahead is $j$. Clock time is a label:
    interval $k$ starts at midnight plus $k\Delta t$, where
    $\Delta t=0.5$ h.

    We use absolute-time indices: $\widehat T_{\tau|k}$ is the temperature
    predicted for boundary $\tau$ using the information available at $k$.
    The index before the bar gives the target time; the index after the bar
    gives the planning time. Every entry in this plan is calculated at $k$,
    including predictions for future target times.

    - $k$ is the current interval: 0 at 00:00, 1 at 00:30 and 12 at 06:00.
    - $j$ counts intervals ahead: $0,\ldots,N-1$ for powers and $0,\ldots,N$ for temperatures.
    - $\tau=k+j$ is the absolute target interval or boundary.
    - $P_{\tau|k}$ is heating power for interval $\tau$, planned at interval $k$.
    - $\widehat T_{\tau|k}$ is predicted indoor temperature at boundary $\tau$.
    - $\widehat T_{o,\tau|k}$ is the outdoor input: a nowcast for $\tau=k$ and a forecast
      for $\tau>k$.
    - $T_k^{\mathrm{meas}}$ is the indoor reading available at the start of interval $k$.
    - $p_{\tau}$ is the given price for interval $\tau$.
    - $L^{\mathrm{end}}_\tau,U^{\mathrm{end}}_\tau$ are the given comfort bounds
      at the end of interval $\tau$, applied to temperature at boundary $\tau+1$.

    Prices and comfort schedules over the horizon are announced inputs available
    at $k$. Weather forecasts for future target times are also available at $k$;
    the future realised weather is used later for evaluation. The predicted indoor
    trajectory depends on the candidate heating actions. The hat marks a
    prediction or forecast; it gives no guarantee of accuracy.

    For example, at $k=12$ (06:00), the accepted plan contains

    $$\underbrace{P^*_{12|12}}_{\text{apply from 06:00 to 06:30}},\qquad
    \underbrace{P^*_{13|12}}_{\text{provisional action from 06:30 to 07:00}}.$$

    At 06:30, a new reading becomes available. Compare the earlier prediction
    with the starting value of the new plan:

    $$\underbrace{\widehat T_{13|12}}_{\substack{\text{target: 06:30}\\\text{information available at 06:00}}},\qquad
    \underbrace{\widehat T_{13|13}=T_{13}^{\mathrm{meas}}}_{\substack{\text{starting value at 06:30}\\\text{from the new measurement}}}.$$

    Both quantities refer to 06:30. The earlier prediction can differ from the
    new reading because the model and available inputs are approximate. The new
    plan starts from the measurement and predicts $\widehat T_{14|13}$ for 07:00.
    The hat identifies an entry in the predicted trajectory; its starting entry
    is fixed by the current reading.

    **Problem (MPC).** At each interval $k$, solve

    $$\begin{aligned}
    \min_{P,\widehat T,s}\quad &\Delta t\sum_{\tau=k}^{k+N-1}
        \left[p_{\tau}P_{\tau|k}+\gamma s_{\tau|k}\right]\\
    \mathrm{s.t.}\quad &\widehat T_{k|k}=T_k^{\mathrm{meas}},\\
    &\widehat T_{\tau+1|k}=(1-b)\widehat T_{\tau|k}
       +b\widehat T_{o,\tau|k}+c\,\mathrm{COP}\,P_{\tau|k},\\
    &0\le P_{\tau|k}\le P_{\max},\quad s_{\tau|k}\ge0,\\
    &\widehat T_{\tau+1|k}+s_{\tau|k}\ge L^{\mathrm{end}}_\tau,\quad
      \widehat T_{\tau+1|k}\le U^{\mathrm{end}}_\tau,\qquad \tau=k,\ldots,k+N-1.
    \end{aligned}$$

    The control decisions are the heating powers $P$. The auxiliary
    variables $\widehat T$ represent the predicted states implied by those
    powers and the thermal model; the auxiliary slack variables $s$
    allow a penalised shortfall below the lower comfort bound. The upper bound
    remains hard. Some initial conditions can still make the upper-bound
    constraint infeasible.

    An asterisk marks the solution of an accepted optimisation. For an accepted
    MPC plan, the applied command is

    $$\underbrace{P_k^{\mathrm{act}}}_{\text{command applied now}}
      =\underbrace{P^*_{k|k}}_{\text{first action of the accepted plan}}.$$

    The later $P^*_{\tau|k}$ values are provisional. Task D uses the supplied
    backup policy when a plan is rejected or the current reading is unavailable.
    The implementation also corrects tiny numerical excesses at actuator limits
    within the stated acceptance tolerance. The simulation receives the resulting
    applied command.

    Set $\gamma=25$ £/(°C h). The objective adds predicted electricity cost
    to a comfort penalty. Report the metered bill separately from that assigned
    penalty. Both terms include $\Delta t$. The coefficient
    $c$ already contains the sampling effect, so the thermal
    equation uses $c\,\mathrm{COP}\,P$ without another $\Delta t$. The final
    predicted temperature remains subject to the stated comfort constraints.

    At the next interval, replace the initial measurement, weather inputs, prices
    and comfort bounds. Keep the model coefficients and, in the core example,
    $N=48$ fixed. The CVXPY variables retain their dimensions, so we
    build the problem once and update the input `Parameter` values before each
    solve.
    ''')

    md('state-estimation', r'''
    The example uses the indoor reading directly as the prediction initial
    condition. A controller with hidden states or noisy outputs may instead need
    state estimation. An estimator combines observations and a model under stated
    assumptions. Sensor calibration requires independent information about bias.
    This example initialises predictions directly from the available indoor
    reading; the simulated true state is reserved for evaluation.
    ''')

    md('week7-code-map', r'''
    ### From the Week 7 code to this problem (slides 16, 22)

    Tutorial 7.1 already used `initial_temperature` and `price` Parameters. Here
    we replace their values at each origin and also parameterise the weather and
    comfort vectors. The variable names are shorter, but their units and
    dimensions are unchanged.

    Python arrays use local positions within each plan. At planning time $k$,
    `P[j]`, `T[j]` and `s[j]` represent $P_{\tau|k}$,
    $\widehat T_{\tau|k}$ and $s_{\tau|k}$. In particular,
    `T[0]` stores $\widehat T_{k|k}=T_k^{\mathrm{meas}}$ and `P[0]`
    stores the first action $P_{k|k}$. Their array positions stay zero when
    the planning time advances. Use $\tau=k+j$ to map an array position to an
    absolute target time. `lower_parameter[j]` and `upper_parameter[j]` store
    $L^{\mathrm{end}}_\tau,U^{\mathrm{end}}_\tau$ and constrain `T[j+1]`.

    | Tutorial 7.1 | Tutorial 8.1 | Change |
    |---|---|---|
    | `power` | `P` | $N$ electrical controls in kW; same capacity |
    | `temperature` | `T` | $N+1$ boundary temperatures in °C |
    | `slack` | `s` | $N$ nonnegative lower shortfalls in °C |
    | `initial_temperature` Parameter | `initial_temperature` / `controller['initial']` | Replace its value with the available reading |
    | Fixed `outdoor` array | `outdoor_parameter` | Supply the origin-specific weather vector |
    | `price` Parameter | `price_parameter` | Refresh the announced price slice |
    | `lower`, `upper` arrays | `lower_parameter`, `upper_parameter` | Shift the endpoint requirements |
    | Lower-soft constraints and objective | Same recurrence, bounds and cost-plus-slack form | Specify and solve again at each origin |

    The Week 7 recurrence is $T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k$. The relevant CVXPY pattern
    becomes:

    ```python
    # Tutorial 7.1: w71-task-b, with declarations in w71-parameters
    temperature[1:] == (1-b)*temperature[:-1] + b*outdoor + c*COP*power
    # Tutorial 8.1: weather is now an updated Parameter
    T[1:] == (1-b)*T[:-1] + b*outdoor_parameter + c*COP*P
    ```

    The lower-soft pattern comes from `w71-soft-parameters` and `w71-task-e`.
    Tutorial 7.1 writes its endpoint bounds as $L_k,U_k$; here the same values
    are labelled $L_k^{\mathrm{end}},U_k^{\mathrm{end}}$ to show their timing:
    $T_{k+1}+s_k\ge L_k^{\mathrm{end}}$, $T_{k+1}\le U_k^{\mathrm{end}}$, with $\Delta t\sum_k(p_kP_k+\gamma s_k)$. Its form is
    retained; today's penalty is explicitly $\gamma=25$ £/(°C h). Use the
    local declarations below to run this notebook independently.
    ''')

    md('task-b-text', r'''
    ### Task B — reconstruct one MPC problem (slide 16; 15 min)

    Complete **Problem (MPC)** from the formulation: current-state equality,
    dynamics, power limits, lower-soft/upper-hard comfort and objective. Identify
    the decisions and Parameters. Work for 12 minutes, then use 3 minutes to
    explain the physical role of each constraint. Variable declarations, input
    values and acceptance checks are supplied.

    Build the problem once here. In step 3 of the control loop, update its five
    Parameters and solve it again. The coefficients $b,c$, COP, actuator capacity,
    $N$ and $\gamma$ stay fixed.

    Use the **two-interval heating optimiser** supplied on Moodle
    to explore the feasible set, comfort shortfalls and objective before coding.
    Open the downloaded HTML file in a browser; it works offline.

    For independent study of CVXPY syntax and Parameters, see the
    [CVXPY introduction](https://www.cvxpy.org/tutorial/intro/index.html#parameters).
    ''')

    code('prediction-parts', r'''
    # Familiar Week 7 LP parts. Build once; update Parameter values at each origin.
    P=cp.Variable(N,name='electrical_heating_kw')
    T=cp.Variable(N+1,name='predicted_temperature_c')
    s=cp.Variable(N,nonneg=True,name='lower_comfort_slack_c')
    initial_temperature=cp.Parameter(name='available_indoor_c')
    outdoor_parameter=cp.Parameter(N,name='available_outdoor_c')
    price_parameter=cp.Parameter(N,nonneg=True,name='price_gbp_per_kwh')
    lower_parameter=cp.Parameter(N,name='lower_c')
    upper_parameter=cp.Parameter(N,name='upper_c')
    initial_inputs={'origin':first_origin,'measurement_c':first_view['measurement_c'],'outdoor':outdoor_vector,
        'price':first_schedule.import_price_gbp_per_kwh.to_numpy(),
        'lower':first_schedule.comfort_min_c.to_numpy(),'upper':first_schedule.comfort_max_c.to_numpy(),
        'valid_time':pd.DatetimeIndex(first_schedule.index),'provenance':forecast_alignment}
    initial_temperature.value=initial_inputs['measurement_c']
    outdoor_parameter.value=initial_inputs['outdoor']
    price_parameter.value=initial_inputs['price']
    lower_parameter.value=initial_inputs['lower']
    upper_parameter.value=initial_inputs['upper']
    ''')

    task('B', r'''
    constraints=[T[0]==initial_temperature,
        T[1:]==(1-b)*T[:-1]+b*outdoor_parameter+c*COP*P,
        P>=0,
        P<=P_MAX,
        T[1:]+s>=lower_parameter,
        T[1:]<=upper_parameter]
    objective=cp.Minimize(dt*cp.sum(cp.multiply(price_parameter,P)+gamma*s))
    problem=cp.Problem(objective,constraints)
    ''', r"""# s is already declared nonnegative in the supplied variables.
# TODO B: complete constraints and objective from the mathematical formulation.
# constraints=[ ... ]
# objective=cp.Minimize( ... )
# problem=cp.Problem(objective,constraints)
raise NotImplementedError('Task B: formulate the repeated heating problem.')""", 'code')

    code('first-solve', r"""controller={'problem':problem,'P':P,'T':T,'s':s,'initial':initial_temperature,
    'outdoor':outdoor_parameter,'price':price_parameter,'lower':lower_parameter,'upper':upper_parameter,
    'N':N,'model':model,'gamma':gamma,'comfort_margin_c':comfort_margin_c}
first_plan=solve_update(controller,initial_inputs)""")

    response('B', 'Which quantities change at the next origin?', textwrap.dedent(r'''
    The available measurement replaces the initial-state equality; the latest
    forecasts and shifted schedules replace the horizon inputs. $P,T,s$
    are optimised again. The supplied $b,c,\mathrm{COP}$ and penalty remain fixed.
    The update recomputes the operating plan using those fixed coefficients.
    ''').strip())

    md('task-c-text', r'''
    ### Task C — trace two updates by hand (slide 18; 8 min)

    This **hand calculation** uses a three-interval prediction horizon with $b_{toy}=0.1$ and $(c\,\mathrm{COP})_{toy}=0.2$
    °C/kW. These illustrative coefficients apply to the hand calculation; the full
    controller retains the supplied course model.

    At 00:00, the measured temperature is 20°C, the first outdoor value is 10°C
    and a provisional plan is [2, 4, 1] kW. Applying 2 kW gives the first
    prediction $0.9(20)+0.1(10)+0.2(2)=19.4$°C. At 00:30, the actual sensor reading is 19.0°C
    and the newly optimised plan is [3, 1, 0] kW. Its first outdoor input is again
    10°C. At 01:00 the next measurement is 19.1°C. Use these supplied plans for
    the hand calculation.

    Use 5 minutes for the calculation and 3 minutes for discussion.

    Use the same workflow for the first two updates. In step 3, initialise the
    prediction at the current reading. In step 4, select the new plan's first
    action. The reference building continues from its current physical state.
    Complete the table below and identify the discarded action. Each toy plan has
    three actions and four temperature boundaries; the core case has 48 actions.
    ''')

    task('C', r'''
    | Origin | Prediction initial condition (°C) | Applied action (kW) | Predicted next T (°C) | Next observed T (°C) |
    |---|---:|---:|---:|---:|
    | 00:00 | 20.0 | 2 | 19.4 (given) | 19.0 |
    | 00:30 | 19.0 | 3 | $0.9(19)+0.1(10)+0.2(3)=18.7$ | 19.1 |

    At 00:30 the prediction starts from the 19.0°C reading; the earlier prediction
    was 19.4°C. Apply the new first action of 3 kW in place of the old second
    action of 4 kW. At 01:00 initialise the next prediction at 19.1°C. Both
    three-interval horizons keep three actions and four state boundaries; their
    endpoints move from 01:30 to 02:00.
    ''', r"""| Origin | Prediction initial condition (°C) | Applied action (kW) | Predicted next T (°C) | Next observed T (°C) |
|---|---:|---:|---:|---:|
| 00:00 | 20.0 | 2 | 19.4 (given) | 19.0 |
| 00:30 | TODO | TODO | TODO | 19.1 |

TODO: state the prediction initial condition at 01:00, identify the discarded old action and
give the endpoint of each three-interval horizon.""", 'markdown')

    md('reference-text', r'''
    ## 3. Apply one action and repeat (slides 19–27)

    The prediction model plans the heating. A separate **synthetic building
    simulation**, called the reference model, produces the temperature after we
    apply an action. It includes air and fabric temperatures, heat gains, varying
    COP, process noise and sensor noise. Both true initial temperatures are 20°C.

    The controller receives one indoor measurement at reset and one after each
    applied action. The underlying air and fabric temperatures are retained for
    evaluation only. The controller receives the sensor reading. A new measurement
    sets the starting point of the next prediction without resetting the simulated
    building.

    This is a synthetic test. In practice, we could test a controller using
    measurements from a real building or a detailed physics-based model such as
    EnergyPlus. Here the richer simulation lets us examine what happens when the
    simple prediction model does not describe the building exactly.
    ''')

    code('reference-diagram', "show_concept('C81-05','19')")

    code('first-plan-figure', r"""if not first_plan['accepted']:raise RuntimeError('Inspect the input/solve report before continuing.')
fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
axes[0].stairs(first_plan['power_kw'],first_plan['state_time'],color=COLOURS['MPC'],linestyle='--')
axes[0].set(ylabel='Proposed HP power (kW)',xlabel='Interval time')
axes[1].plot(first_plan['state_time'],first_plan['temperature_c'],color=COLOURS['MPC'],linestyle='--',label='Predicted')
comfort_lines(axes[1],first_plan['valid_time'],first_plan['lower'],first_plan['upper'])
axes[1].scatter(first_origin,first_plan['temperature_c'][0],color='black',label='Measured initial condition',zorder=5)
axes[1].fill_between(first_plan['state_time'][1:],first_plan['lower'],first_plan['upper'],color='tab:green',alpha=.12,label='Comfort band')
axes[1].set(ylabel='Predicted temperature (°C)',xlabel='Boundary time');axes[1].legend(fontsize=9)
for ax in axes:
    clock_axis(ax);ax.axvspan(first_origin,first_origin+pd.Timedelta(hours=dt),color='tab:orange',alpha=.25,label='First applied interval')
axes[0].legend(fontsize=9)
fig.suptitle(f'Issue {first_origin:%d %b %H:%M} | N=48 | Δt=0.5 h | γ=25 £/(°C h)',fontsize=12)
save_figure(fig,'F81-01')""")

    md('guided-trace-text', r'''
    ### Follow two MPC updates (slide 23)

    The **MPC step-through player** supplied on Moodle lets you
    reveal a plan, apply its first action and inspect the returned observation.
    Open the downloaded HTML file in a browser; it works offline.

    Run the six-step workflow twice, using the Task B optimisation problem and
    a fresh building simulation. Each plan still contains 48 intervals. Keep this
    pseudocode beside the code as a reminder:

    ```text
    Create a fresh building simulation; reuse the Task B problem.
    for k = 0, ..., number_of_updates - 1:
        1. Read measurement: read the stored indoor temperature.
        2. Gather inputs: get the nowcast, forecasts and schedules.
        3. Optimise and check: update Parameters, solve and validate.
        4. Select action: take the accepted plan's first move.
        5. Apply and observe: apply once and store the next measurement.
        6. Repeat: use that measurement at the next origin.
    ```

    Here `number_of_updates = 2`. Before running the cell, predict the origin of
    the second row and which temperature initialises its prediction.
    `SimulationSession` stores observations and evolves the reference building;
    `solve_update` solves and checks the Task B problem. This short demonstration
    stops if a plan is rejected. Task D adds the supplied backup handling.
    ''')

    code('guided-two-updates', r'''
    trace_session=SimulationSession(operation,model,DAY,M=2,seed=SEED)
    trace_controller=controller  # reuse the problem built in Task B
    trace_records=[]

    for k in range(2):
        # 1. Read measurement.
        view=trace_session.available()
        measured_temperature=view['measurement_c']
        # 2. Gather inputs.
        inputs=trace_session.inputs(announced,forecasts,N)
        # 3. Optimise and check.
        trace_controller['initial'].value=measured_temperature
        for name in ['outdoor','price','lower','upper']:
            trace_controller[name].value=inputs[name]
        plan=solve_update(trace_controller,inputs)
        if not plan['accepted']:
            raise RuntimeError('Trace stopped: '+plan['reason'])
        # 4. Select action.
        action_kw=first_action(plan)
        # 5. Apply and observe.
        next_measurement=trace_session.advance(action_kw,plan)
        trace_records.append({'origin':view['origin'],'measured_initial_c':measured_temperature,
            'first_valid_time':inputs['valid_time'][0],'last_valid_time':inputs['valid_time'][-1],
            'N':N,'accepted':plan['accepted'],'status':plan['status'],'fallback':False,'abort':False,'reason':'',
            'applied_power_kw':trace_session.intervals[-1]['applied_power_kw'],
            'next_predicted_c':plan['temperature_c'][1],'next_measurement_c':next_measurement})
        # 6. Repeat.

    guided_trace=pd.DataFrame(trace_records)
    display(guided_trace[['origin','measured_initial_c','N','applied_power_kw',
                         'next_predicted_c','next_measurement_c']].round(3))
    ''')

    md('guided-next-origin', r'''
    The second row starts from the measurement returned by the first update.
    Compare it with the earlier one-step prediction. The origin moves from 00:00
    to 00:30, while each plan retains 48 intervals. Each call to `advance` applies
    one action and produces the reading used at the next iteration.
    ''')

    code('guided-next-update', r"""# Supplied export of the two updates above; no new simulation step.
export_table(guided_trace,'guided_update_trace.csv')
export_run(trace_session.result('Guided trace'),'guided_trace')""")

    md('loop-text', r'''
    ### Task D — implement the MPC loop (slide 24; 22 min)

    Extend the two-update example to `number_of_updates = M = 48`. TODOs D1–D5
    match steps 1–5; the `for` loop supplies step 6. Use the same workflow:

    ```text
    Create a fresh building simulation; reuse the Task B problem.
    for k = 0, ..., number_of_updates - 1:
        1. Read measurement: read the stored indoor temperature.
        2. Gather inputs: get the nowcast, forecasts and schedules.
        3. Optimise and check: update Parameters, solve and validate.
        4. Select action: take the accepted plan's first move.
        5. Apply and observe: apply once and store the next measurement.
        6. Repeat: use that measurement at the next origin.
    ```

    In step 3, update the initial temperature, outdoor temperature, price, lower
    bound and upper bound Parameters of the Task B problem.

    Keep the supplied failure and logging code. For step 4, an accepted plan
    supplies its first action; the `else` branch selects backup heating when
    needed. An abort ends the loop before step 5. Otherwise, step 5 advances the
    building once and stores the next reading for step 1 of the next iteration.

    Allow 4 minutes for steps 1–2, 5 for step 3, 5 for steps 4–5, 4 for checking
    48 actions and 49 state boundaries, and 4 for discussion.
    ''')

    task('D', r'''
    session=SimulationSession(operation,model,DAY,M=M,seed=SEED)
    policy=FallbackPolicy()
    last_command=None
    stored_plans=[]
    for k in range(M):
        started=time.perf_counter()
        # 1. Read measurement.
        view=session.available()
        measured_temperature=view['measurement_c']
        if not np.isfinite(measured_temperature):
            plan=rejected_plan('Missing current temperature.',origin=view['origin'])
        else:
            try:
                # 2. Gather inputs.
                inputs=session.inputs(announced,forecasts,controller['N'])
                # 3. Optimise and check.
                controller['initial'].value=measured_temperature
                for name in ['outdoor','price','lower','upper']:
                    controller[name].value=inputs[name]
                plan=solve_update(controller,inputs)
            except Exception as error:
                plan=rejected_plan(error,'invalid_inputs',view['origin'])
        # 4. Select action.
        if plan['accepted']:
            action_kw=first_action(plan)  # the copied, checked P[0]
            decision={'action_kw':action_kw,'fallback':False,'abort':False,'reason':''}
            policy.consecutive_missing=0
        else:
            decision=choose_command(plan,measured_temperature,view['lower_c'],view['upper_c'],P_MAX,policy,last_command)
        if decision['abort']:
            session.abort(decision['reason'])
            break
        if plan['accepted'] and view['origin'].hour in (0,6,12) and view['origin'].minute==0:
            stored_plans.append(copy.deepcopy(plan))  # before applying the action
        # 5. Apply and observe.
        next_measurement=session.advance(decision['action_kw'],plan,decision,time.perf_counter()-started)
        last_command=session.intervals[-1]['applied_power_kw']
        # 6. Repeat.
    mpc_day=session.result('MPC')
    mpc_day['plans']=stored_plans
    print('Applied intervals:',len(mpc_day['intervals']),'| State boundaries:',len(mpc_day['boundaries']))
    ''', r'''
    session=SimulationSession(operation,model,DAY,M=M,seed=SEED)
    policy=FallbackPolicy()
    last_command=None
    stored_plans=[]
    for k in range(M):
        started=time.perf_counter()
        # 1. Read measurement.
        view=session.available()
        # TODO D1: measured_temperature = view['measurement_c']
        raise NotImplementedError('Task D: complete the six-step MPC loop.')
        if not np.isfinite(measured_temperature):
            plan=rejected_plan('Missing current temperature.',origin=view['origin'])
        else:
            try:
                # 2. Gather inputs.
                # TODO D2: inputs = session.inputs(announced, forecasts, controller['N'])
                # 3. Optimise and check.
                # TODO D3a: set controller['initial']; update outdoor, price, lower, upper.
                # TODO D3b: plan = ... (use solve_update).
                pass
            except Exception as error:
                plan=rejected_plan(error,'invalid_inputs',view['origin'])
        # 4. Select action.
        if plan['accepted']:
            # TODO D4: select action_kw from the accepted first move.
            decision={'action_kw':action_kw,'fallback':False,'abort':False,'reason':''}
            policy.consecutive_missing=0
        else:
            decision=choose_command(plan,measured_temperature,view['lower_c'],view['upper_c'],P_MAX,policy,last_command)
        if decision['abort']:
            session.abort(decision['reason'])
            break
        if plan['accepted'] and view['origin'].hour in (0,6,12) and view['origin'].minute==0:
            stored_plans.append(copy.deepcopy(plan))  # before applying the action
        # 5. Apply and observe.
        # TODO D5: call session.advance once; store its return in next_measurement.
        last_command=session.intervals[-1]['applied_power_kw']
        # 6. Repeat.
    mpc_day=session.result('MPC')
    mpc_day['plans']=stored_plans
    print('Applied intervals:',len(mpc_day['intervals']),'| State boundaries:',len(mpc_day['boundaries']))
    ''')

    code('successive-plans', r"""logs=mpc_day['intervals'];boundaries=mpc_day['boundaries']
fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
axes[0].plot(first_plan['state_time'],first_plan['temperature_c'],color='tab:purple',ls='--',label='Indoor prediction made at 00:00')
axes[0].plot(boundaries.state_time,boundaries.true_temperature_c,color=COLOURS['MPC'],lw=2,label='Realised MPC temperature')
axes[0].plot(logs.interval_end,logs.predicted_next_c,'.',color='tab:orange',ms=3,label='New MPC one-step predictions')
comfort_lines(axes[0],reporting_times,announced.loc[reporting_times].comfort_min_c,announced.loc[reporting_times].comfort_max_c)
axes[1].stairs(first_plan['power_kw'],first_plan['state_time'],color='tab:purple',ls='--',label='Heating planned at 00:00')
axes[1].stairs(logs.applied_power_kw,state_time,color=COLOURS['MPC'],lw=1.5,label='First moves actually applied by MPC')
for ax in axes:clock_axis(ax);ax.set_xlabel('Time');ax.legend(fontsize=8)
axes[0].set_ylabel('Indoor temperature (°C)');axes[1].set_ylabel('Electrical heating (kW)')
save_figure(fig,'F81-02')""")

    code('moving-plan-frames', r"""from PIL import Image
moving_frames=[]
moving_frame_manifest=[]
for frame_index,stored in enumerate(mpc_day['plans']):
    origin=stored['origin'];past=boundaries.loc[boundaries.state_time<=origin]
    applied=logs.loc[logs.interval_start<origin]
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    axes[0].plot(past.state_time,past.measured_temperature_c,color='black',label='Available past readings')
    axes[0].plot(stored['state_time'],stored['temperature_c'],ls='--',label='New predicted indoor T')
    axes[0].scatter(origin,stored['temperature_c'][0],color='black',zorder=4,label='Current measured initial state')
    comfort_lines(axes[0],stored['valid_time'],stored['lower'],stored['upper'])
    if len(applied):axes[1].stairs(applied.applied_power_kw,boundaries.state_time.iloc[:len(applied)+1],color='black',label='Already applied')
    axes[1].stairs(stored['power_kw'],stored['state_time'],ls='--',label='New provisional plan')
    axes[1].plot(stored['state_time'][:2],[stored['power_kw'][0]]*2,color='tab:orange',lw=5,label='Only next action applied')
    for ax in axes:
        ax.set_xlim(state_time[0],state_time[-1]+pd.Timedelta(hours=12))
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,6,12,18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %Hh'));ax.set_xlabel('Day / hour');ax.legend(fontsize=8)
    axes[0].set_ylim(16,26);axes[1].set_ylim(-.1,5.4)
    axes[0].set_ylabel('Indoor temperature (°C)');axes[1].set_ylabel('Electrical heating (kW)')
    fig.suptitle(f'Origin {origin:%d %b %H:%M}: 48-step horizon, first action held for 30 min',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    if EXPORT_ARTEFACTS:
        for ext in ['png','svg']:fig.savefig(FIGURE_DIR/f'w8d1_moving_plan_{frame_index:02d}.{ext}',dpi=220)
        moving_frames.append(Image.open(FIGURE_DIR/f'w8d1_moving_plan_{frame_index:02d}.png').convert('RGB'))
        moving_frame_manifest.append({'frame':frame_index,'origin':str(origin),'N':N,'M':M,'seed':SEED,
            'producer':'w81-moving-plan-frames','png':f'w8d1_moving_plan_{frame_index:02d}.png',
            'svg':f'w8d1_moving_plan_{frame_index:02d}.svg','evidence_prefix':'successive_plans',
            'assumptions':'Stored observation/plan before first action; future proposed, past applied.',
            'source_sha256':json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},sort_keys=True)})
    plt.show();plt.close(fig)
if EXPORT_ARTEFACTS:
    moving_frames[0].save(FIGURE_DIR/'w8d1_moving_horizon.gif',save_all=True,append_images=moving_frames[1:],duration=1500,loop=0)
    export_table(pd.DataFrame(moving_frame_manifest),'moving_plan_frame_manifest.csv')""")

    md('errors', r"""In the first plot, the dashed curve is the indoor-temperature prediction made
at midnight. The blue curve shows the temperature during MPC operation.
Orange points show the one-step predictions made immediately before each
action. The prediction and simulated temperature can differ because the
thermal model is approximate, future weather is uncertain and readings contain
noise. The next section adds the temperature obtained by following the whole
midnight plan.

The separate plan frames show the solves made at 00:00, 06:00 and 12:00. Each
frame starts its prediction at the measurement available then, extends 48
intervals ahead and highlights the first heating action. MPC still updates
every 30 minutes; these are three selected views of that repeated loop.

The calculation below compares the weather forecasts used by MPC with the
weather that subsequently occurred. It is a performance check after the run;
future realised weather was not supplied to the controller. The stored indoor
prediction error is $r_{k+1}=T^{\mathrm{meas}}_{k+1}-\widehat T^*_{k+1|k}$.
It measures disagreement between a prediction and the next reading, without
identifying which error source caused it.""")

    code('forecast-error-evidence', r"""# Supplied diagnostic: score future forecasts actually used within the reporting day.
used_forecasts=mpc_day['forecast_records'].query('lead_steps >= 1').copy()
used_forecasts=used_forecasts.loc[used_forecasts.valid_time<state_time[-1]].copy()
used_forecasts['realised_outdoor_c']=operation.loc[pd.DatetimeIndex(used_forecasts.valid_time),'outdoor_temperature_c'].to_numpy()
used_forecasts['actual_minus_forecast_c']=used_forecasts.realised_outdoor_c-used_forecasts.outdoor_input_c
forecast_error_summary=pd.DataFrame([{'quantity':'Outdoor forecast error: actual minus forecast',
    'scoring':'Origin–lead pairs; j>=1; valid times in reporting day',
    'pairs':len(used_forecasts),'rmse_c':np.sqrt(np.mean(used_forecasts.actual_minus_forecast_c**2)),
    'mean_error_c':used_forecasts.actual_minus_forecast_c.mean()}])
display(forecast_error_summary)
export_table(used_forecasts,'used_forecast_errors.csv')
export_table(forecast_error_summary,'used_forecast_error_summary.csv')""")

    code('error-diagram', "show_concept('C81-06','26,27,31')")

    md('fixed-text', r'''
    ## 4. Fixed operation, MPC and the thermostat (slides 28–36)

    ### Task E — fixed plan versus MPC (slide 29; 10 min)

    The fixed optimisation schedule determines all 48 actions once, using the
    first observation, first forecast and the same lower-soft problem as MPC. It
    then follows that initial sequence without revision. The weather forecast now
    has error, and the reference dynamics differ from the fitted $1R1C$
    prediction model used in planning. MPC uses new measurements, refreshed
    forecasts and a moving horizon endpoint. The comparison changes these three
    features jointly, so it cannot isolate the effect of feedback alone.

    Both runs start independent reference models at the same true air/fabric
    temperatures, use the same realised weather, tariff, comfort bounds, capacity
    and paired noise innovations. Their trajectories can diverge after different
    actions. The supplied runner produces the fixed-policy comparison. The MPC
    result comes from your visible loop.

    Calculate the five quantities below from the supplied log columns. Use 6
    minutes for calculation and helper cross-checks, then 4 minutes to discuss one
    observation and one limitation. Interpret cost and comfort separately.

    Use actual applied electrical power and the simulated temperature at the end
    of each interval. Calculate:

    - **Heating electricity:** $E=\Delta t\sum_k P_k^{\mathrm{act}}$ in kWh.
    - **Electricity bill:** $C=\Delta t\sum_k p_kP_k^{\mathrm{act}}$ in £.
    - **Peak heating power:** $P^{\mathrm{peak}}=\max_k P_k^{\mathrm{act}}$ in kW.
    - **Below-band discomfort:** $D^-=\Delta t\sum_k\max(L_k^{\mathrm{end}}-T_{k+1}^{\mathrm{true}},0)$ in °C h.
    - **Above-band discomfort:** $D^+=\Delta t\sum_k\max(T_{k+1}^{\mathrm{true}}-U_k^{\mathrm{end}},0)$ in °C h.

    Predicted slack belongs to the optimisation. The electricity bill includes the
    cost of applied heating only. The comfort metrics use sampled endpoint
    temperatures, so temperatures between endpoints require a separate assessment.

    All five input vectors have $M=48$ entries: applied power (kW),
    interval price (£/kWh), true **endpoint** temperature (°C), and the original
    lower/upper endpoint limits (°C). Boundary 0 is excluded from comfort scoring.
    The short fixture first checks the same expressions on two intervals. Supplied
    code retains completion/fallback metadata and the remaining detailed metrics.
    ''')

    code('reviewed-runner-import', r"""# Introduced only after Task D: a reviewed reusable runner for later comparisons.
from course_utils.week8_control import run_experiment, STRATEGIES
fixed_day=run_experiment('Fixed schedule',operation,forecasts,model,day=DAY,M=M,N=N,
    seed=SEED,gamma=gamma,comfort_margin_c=comfort_margin_c)
paired_runs={'Fixed schedule':fixed_day,'MPC':mpc_day}""")

    code('metric-inputs', r"""METRIC_KEYS=['energy_kwh','bill_gbp','peak_kw','below_degree_hours','above_degree_hours']
metric_fixture=pd.DataFrame({'applied_power_kw':[2.,0.], 'price_gbp_per_kwh':[.2,.4],
    'true_end_c':[19.,23.], 'lower_c':[20.,20.], 'upper_c':[22.,22.]})
metric_input_columns=['applied_power_kw','price_gbp_per_kwh','true_end_c','lower_c','upper_c']
display(metric_fixture)
display(mpc_day['intervals'][['interval_start','interval_end']+metric_input_columns].head(3))
print('Full-day vectors:',{k:mpc_day['intervals'][k].shape for k in metric_input_columns})""")

    task('E', r'''
    def manual_core_metrics(power,price,true_endpoint,lower,upper,dt):
        shortfall=np.maximum(lower-true_endpoint,0)
        overshoot=np.maximum(true_endpoint-upper,0)
        return {'energy_kwh':float(dt*np.sum(power)),
                'bill_gbp':float(dt*np.dot(price,power)),
                'peak_kw':float(np.max(power)),
                'below_degree_hours':float(dt*shortfall.sum()),
                'above_degree_hours':float(dt*overshoot.sum())}

    fixture_metrics=manual_core_metrics(*(metric_fixture[k].to_numpy() for k in metric_input_columns),dt)
    display(pd.Series(fixture_metrics))
    ''', r"""def manual_core_metrics(power,price,true_endpoint,lower,upper,dt):
    # TODO E: return energy_kwh, bill_gbp, peak_kw, below_degree_hours,
    # and above_degree_hours using the five displayed expressions.
    raise NotImplementedError('Task E: calculate applied-power and true-endpoint metrics.')

fixture_metrics=manual_core_metrics(*(metric_fixture[k].to_numpy() for k in metric_input_columns),dt)
display(pd.Series(fixture_metrics))""")

    code('metric-cross-check', r"""def manual_metrics_for_run(result):
    frame=result['intervals']
    if (not result['complete'] or result['fixed_plan_failed'] or len(frame)==0
            or len(frame)!=result['reporting_intervals']):
        return dict.fromkeys(METRIC_KEYS,np.nan)
    return manual_core_metrics(*(frame[k].to_numpy() for k in metric_input_columns),result['dt'])

manual_metric_records=[]
for name,result in paired_runs.items():
    calculated=manual_metrics_for_run(result)
    reviewed=realised_metrics(result)
    np.testing.assert_allclose([calculated[k] for k in METRIC_KEYS],
                               [reviewed[k] for k in METRIC_KEYS],rtol=0,atol=1e-10,equal_nan=True)
    manual_metric_records.append({'strategy':name,**calculated})
paired_metrics=pd.DataFrame([realised_metrics(result) for result in paired_runs.values()])
display(paired_metrics[['strategy',*METRIC_KEYS,'mean_endpoint_temperature_c',
                       'final_temperature_c','complete','fallback_count']].round(4))
export_table(pd.DataFrame(manual_metric_records),'manual_metric_cross_check.csv')""")

    code('fixed-comparison', r"""fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
axes[0].plot(first_plan['state_time'],first_plan['temperature_c'],color='tab:purple',ls='--',label='Indoor prediction made at 00:00')
for strategy,result in paired_runs.items():
    frame=result['intervals'];states=result['boundaries']
    axes[0].plot(states.state_time,states.true_temperature_c,label='Realised '+strategy,color=COLOURS[strategy])
    axes[1].stairs(frame.applied_power_kw,state_time,label=strategy,color=COLOURS[strategy])
comfort_lines(axes[0],reporting_times,announced.loc[reporting_times].comfort_min_c,announced.loc[reporting_times].comfort_max_c)
for ax in axes:clock_axis(ax);ax.set_xlabel('Time');ax.legend(fontsize=8)
axes[0].set_ylabel('Indoor temperature (°C)');axes[1].set_ylabel('Applied electrical P (kW)')
save_figure(fig,'F81-03')
export_table(paired_metrics,'fixed_vs_mpc_metrics.csv')""")

    code('fixed-comparison-table', r"""pair_table=pd.DataFrame({'Metric':['Bill (£)','Below band (°C h)','Above band (°C h)'],
    **{name:paired_metrics.set_index('strategy').loc[name,['bill_gbp','below_degree_hours','above_degree_hours']].to_numpy(dtype=float) for name in paired_runs}})
display(pair_table)
fig,ax=plt.subplots(figsize=(12.5,3.5))
draw_table(ax,pair_table,fontsize=13,widths=[.46,.27,.27],scale=2)
save_figure(fig,'F81-03-table')
export_table(pair_table,'fixed_vs_mpc_native_table.csv')""")

    response('E', 'State one observed trade-off and one limitation.', textwrap.dedent(r'''
    For this day, MPC lowers the heating bill from £4.591 to £3.459, while
    lower-band deviation increases from 0.127 to 0.652 °C h. Both runs complete
    without fallback. Both realised trajectories fall below the lower comfort
    bound during this day. The hard upper limit constrains predictions, while
    lower shortfalls are permitted through slack. Errors can also move realised
    temperatures away from the predicted band. The comparison measures the joint
    effect of state feedback, updated forecasts and a moving horizon endpoint. The
    result applies to this simulated day and its supplied error settings.
    ''').strip())

    md('thermostat-text', r'''
    ### Task F — add the Week 7 thermostat (slide 33; 8 min)

    The thermostat applies $P_{\max}$ when the available measurement is
    below $R_k^{\mathrm{end}}=(L_k^{\mathrm{end}}+U_k^{\mathrm{end}})/2$, and zero otherwise. Equality switches off, giving
    zero hysteresis. It uses the same sensor channel as MPC. Its heating decision
    depends on the current reading and midpoint, independently of prices and
    future plans. Its midpoint preference differs from the optimisation comfort
    band, so compare mean and final temperature as well as cost.

    Complete the short decision function and check readings below, equal to and
    above the midpoint. The supplied runner uses your function to add a third
    policy. Interpret electricity, bill, peak, both comfort deviations and
    mean/final temperature. Use 5 minutes for code and analysis and 3 minutes for
    debrief. Report these metrics for the heat pump; the equipment boundary
    excludes other site demand.
    ''')

    task('F', r'''
    def thermostat_decision(measurement,lower,upper,pmax):
        midpoint=(lower+upper)/2
        if measurement<midpoint:
            return float(pmax)
        return 0.

    thermostat_thresholds=[thermostat_decision(t,18.,22.,P_MAX) for t in [19.,20.,21.]]
    ''', r"""def thermostat_decision(measurement,lower,upper,pmax):
    # TODO F: below midpoint -> pmax; equality or above -> zero.
    raise NotImplementedError('Task F: implement the midpoint thermostat decision.')

thermostat_thresholds=[thermostat_decision(t,18.,22.,P_MAX) for t in [19.,20.,21.]]""")

    code('thermostat-run', r"""# Supplied plumbing: one reset, one observation per step, the reviewed missing-data policy.
def run_local_thermostat(decision_function,operation,model,M=48,seed=88,missing_readings=()):
    session=SimulationSession(operation,model,DAY,M=M,seed=seed,missing_readings=missing_readings)
    policy=FallbackPolicy();last=None
    for k in range(M):
        started=time.perf_counter();view=session.available();plan=None
        if np.isfinite(view['measurement_c']):
            policy.consecutive_missing=0
            decision={'action_kw':decision_function(view['measurement_c'],view['lower_c'],view['upper_c'],session.capacity),
                      'fallback':False,'abort':False,'reason':''}
        else:
            plan=rejected_plan('Missing current temperature.',origin=view['origin'])
            decision=policy.decide(view['measurement_c'],view['lower_c'],view['upper_c'],session.capacity,last,'Missing observation')
        if decision['abort']:
            session.abort(decision['reason']);break
        session.advance(decision['action_kw'],plan,decision,time.perf_counter()-started)
        last=session.intervals[-1]['applied_power_kw']
    return session.result('Thermostat')

np.testing.assert_array_equal(thermostat_thresholds,[P_MAX,0.,0.])
display(pd.DataFrame({'Reading (°C)':[19.,20.,21.],'Midpoint (°C)':[20.]*3,
                      'Action (kW)':thermostat_thresholds}))
thermostat_day=run_local_thermostat(thermostat_decision,operation,model,M,SEED)
comparison_runs={**paired_runs,'Thermostat':thermostat_day}
comparison=pd.DataFrame([realised_metrics(result) for result in comparison_runs.values()])
display(comparison[['strategy',*METRIC_KEYS,'mean_endpoint_temperature_c',
                    'final_temperature_c','complete']].round(4))""")

    code('thermostat-comparison', r"""metric_columns=['energy_kwh','bill_gbp','peak_kw','below_degree_hours','above_degree_hours',
    'mean_endpoint_temperature_c','final_temperature_c']
metric_labels=['HP electricity (kWh)','HP bill (£)','HP interval peak (kW)','Below band (°C h)',
    'Above band (°C h)','Mean endpoint temperature (°C)','Final temperature (°C)']
compact=pd.DataFrame({'Metric':metric_labels,**{strategy:comparison.set_index('strategy').loc[strategy,metric_columns].to_numpy(dtype=float) for strategy in comparison_runs}})
export_table(comparison,'three_policy_metrics.csv');export_table(compact,'three_policy_metrics_native_table.csv')
fig,ax=plt.subplots(figsize=(12.5,3.5));draw_table(ax,compact,fontsize=12,widths=[.4,.2,.2,.2],scale=1.8)
save_figure(fig,'F81-04')""")

    code('optional-algorithm-export', r"""# Optional authoring asset retained; use the Markdown algorithm in the tutorial.
if EXPORT_ARTEFACTS:
    fig=concept('C81-04')
    for ext in ['png','svg']:fig.savefig(FIGURE_DIR/f'C81-04.{ext}',dpi=220)
    plt.close(fig)
    concept_manifest.append({'id':'C81-04','slides':'14,23,24,36; optional diagram',
        'kind':'schematic; optional authoring asset; Markdown algorithm used in notebook',
        'source':'scripts/week8_day1_visuals.py','svg':'figures/week8_day1/C81-04.svg',
        'source_sha256':hashlib.sha256((ROOT/'scripts/week8_day1_visuals.py').read_bytes()).hexdigest()})""")

    response('F', 'What can the three-policy comparison establish?', textwrap.dedent(r'''
    The thermostat uses 32.5 kWh and costs £8.150, with mean true endpoint
    temperature 21.603°C and 0.088 °C h above the upper bound. MPC costs less but
    operates closer to the lower comfort limit and has more lower-band deviation.
    The thermostat already uses feedback; its different temperature preference and
    lack of price response matter. These results describe one synthetic day with
    an imperfect model, supplied forecast errors and noisy observations. Field
    performance and comparisons across operating conditions require further
    evaluation.
    ''').strip())

    md('daily-cost-text', r"""### Daily totals and interval averages

For interval $k$, electrical energy is $E_k=P_k^{\mathrm{act}}\Delta t$ and
cost is $C_k=p_kE_k$. With half-hour intervals, 2 kW at £0.20/kWh costs
$2\times0.5\times0.20=£0.20$. Multiplying power directly by price would
miss the interval duration.

Over $M=48$ intervals, calculate
$$E_{\mathrm{day}}=\sum_k E_k,\qquad
C_{\mathrm{day}}=\sum_k C_k,\qquad
\overline C_{\mathrm{interval}}=C_{\mathrm{day}}/M.$$
The last quantity is the average cost **per half-hour interval** (£/interval).
Average expenditure per hour is $C_{\mathrm{day}}/(M\Delta t)$ (£/h).
Average paid price is $C_{\mathrm{day}}/E_{\mathrm{day}}$ (£/kWh); it is
weighted by energy use and generally differs from the mean tariff. If no
energy is used, this last ratio is undefined.

The table and cumulative-cost plot below use actual applied heating. Comfort
penalties are excluded from the bill. Compare the total and average cost,
then interpret them alongside the comfort results from Task E.""")

    code('daily-cost', r"""daily_cost_rows=[];daily_interval_rows=[]
fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
for label,run in paired_runs.items():
    rows=run['intervals']
    if not run['complete']:raise RuntimeError('Daily averages require a complete day.')
    energy=rows.applied_power_kw.to_numpy()*dt
    interval_cost=energy*rows.price_gbp_per_kwh.to_numpy()
    total=interval_cost.sum()
    daily_cost_rows.append({'Policy':label,'Energy (kWh)':energy.sum(),
        'Daily bill (GBP)':total,'Mean cost (GBP/interval)':total/len(rows),
        'Mean cost (GBP/h)':total/(len(rows)*dt),
        'Paid price (GBP/kWh)':total/energy.sum() if energy.sum()>0 else np.nan})
    daily_interval_rows.append(pd.DataFrame({'policy':label,'interval_start':rows.interval_start,
        'energy_kwh':energy,'cost_gbp':interval_cost,'cumulative_cost_gbp':np.cumsum(interval_cost)}))
    axes[0].stairs(interval_cost,mdates.date2num(run['boundaries'].state_time),label=label,color=COLOURS[label],baseline=None)
    axes[1].plot(run['boundaries'].state_time,np.r_[0.,np.cumsum(interval_cost)],label=label,color=COLOURS[label])
for ax in axes:
    clock_axis(ax);ax.legend()
axes[0].set(ylabel='Cost per half-hour (GBP)',xlabel='Interval start')
axes[1].set(ylabel='Cumulative electricity bill (GBP)',xlabel='Time')
daily_cost_summary=pd.DataFrame(daily_cost_rows)
daily_cost_intervals=pd.concat(daily_interval_rows,ignore_index=True)
display(daily_cost_summary.round(3))
save_figure(fig,'F81-06',case='Daily cost: interval duration, totals and averages')
export_table(daily_cost_summary,'daily_cost_summary.csv')
export_table(daily_cost_intervals,'daily_cost_intervals.csv')""")

    response('daily-cost', 'Explain the units of the three averages above. Why can two policies face the same tariff but have different average paid prices?', 'Cost per interval divides the bill by 48; cost per hour divides by 24. Paid price divides by consumed kWh. It depends on when each policy uses electricity, so it can differ under the same tariff.')

    md('infrastructure', r'''
    ### Discussion: where should we invest in sensing and communication?

    | Policy | Measurements and information | Communication and computation |
    |---|---|---|
    | Fixed open-loop plan | Initial indoor reading, model, forecast and provided schedules; later readings can monitor operation but do not revise the plan | One initial solve and schedule transfer; local equipment executes the sequence |
    | Midpoint thermostat | Current indoor reading and comfort midpoint; no weather forecast or price input | Frequent local sensor/actuator feedback; simple computation; can operate without a continuous remote connection |
    | MPC | Current measured/estimated indoor state, model, outdoor nowcast, forecast and provided schedules | Repeated optimisation and reliable updates between sensing, controller and BAS; computation can be local or remote |

    All policies still require local actuator controls and basic equipment
    protection. An on-site MPC controller can use local measurements and
    periodically downloaded forecasts.

    For a building with missing, poorly placed or biased indoor sensors, first
    check measurement quality, units, timestamps and actuator operation. Sensor
    calibration or better placement can address systematic measurement errors.
    Identifying a sensor bias requires information beyond repeated use of the same
    reading. Filtering may reduce noise but introduces lag and needs evaluation.
    Once the control-relevant measurements are reliable, assess forecast quality,
    model suitability and supervisory integration against the cost of additional
    sensing and communication.

    **Discuss:** would you invest first in additional sensors, calibrated existing
    sensors, better forecasts or repeated optimisation for this building? State
    which error or operating limitation the investment would address and how you
    would compare cost, comfort and implementation effort. Quantifying the value
    of better measurements requires paired sensor-quality cases, realistic
    equipment costs and evaluation over more days.
    ''')

    code('guided-transfer-exports', r"""trace_inputs=guided_trace[['origin','measured_initial_c','first_valid_time','last_valid_time','N']]
trace_outcomes=guided_trace[['origin','accepted','applied_power_kw','next_predicted_c','next_measurement_c']]
export_table(trace_inputs,'guided_trace_inputs_native_table.csv')
export_table(trace_outcomes,'guided_trace_outcomes_native_table.csv')
asset_specs=[
    ('guided_update_trace.csv',23,'w81-guided-next-update','timestamps; degrees C; kW; counts','Two demonstration updates; N=48'),
    ('guided_trace_inputs_native_table.csv',23,'w81-guided-transfer-exports','timestamps; degrees C; counts','Available inputs at each origin'),
    ('guided_trace_outcomes_native_table.csv',23,'w81-guided-transfer-exports','degrees C; kW','Applied first action and next reading'),
    ('daily_cost_summary.csv',34,'w81-daily-cost','kWh; GBP; GBP/interval; GBP/h; GBP/kWh','Daily totals and distinct averages'),
    ('daily_cost_intervals.csv',34,'w81-daily-cost','kWh; GBP','Actual applied heating; half-hour intervals')]
if EXPORT_ARTEFACTS:
    asset_manifest=pd.DataFrame([{'file':name,'slide':slide,'producer_cell_id':producer,
        'units':units,'assumptions':assumptions,'day':DAY,'seed':SEED,'N':N,'dt_hours':dt,
        'sha256':hashlib.sha256((OUTPUT_DIR/name).read_bytes()).hexdigest(),
        'source_sha256':json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},sort_keys=True)}
        for name,slide,producer,units,assumptions in asset_specs])
    asset_manifest.to_csv(OUTPUT_DIR/'guided_transfer_asset_manifest.csv',index=False)
    asset_manifest.to_csv(ROOT/'teaching/week8_day1_supplemental_placements.csv',index=False)""")

    code('exports', r"""export_table(forecast_alignment,'first_plan_forecast_alignment.csv')
export_table(pd.DataFrame({'interval_start':first_plan['valid_time'],'power_kw':first_plan['power_kw'],
    'outdoor_forecast_c':first_plan['outdoor'],'price_gbp_per_kwh':first_plan['price'],
    'lower_c':first_plan['lower'],'upper_c':first_plan['upper'],'predicted_slack_c':first_plan['slack_c']}),'first_plan_intervals.csv')
export_table(pd.DataFrame({'state_time':first_plan['state_time'],'predicted_temperature_c':first_plan['temperature_c']}),'first_plan_states.csv')
export_run(mpc_day,'successive_plans')
for strategy,result in paired_runs.items():export_run(result,'fixed_vs_mpc_'+strategy.lower().replace(' ','_'))
export_run(thermostat_day,'three_policy_thermostat')
status_report=comparison[['strategy','complete','fallback_count','invalid_solve_count','runtime_s','abort_reason']]
export_table(status_report,'operating_status_and_runtime.csv')
if EXPORT_ARTEFACTS:
    pd.DataFrame(figure_manifest).to_csv(ROOT/'teaching/week8_day1_figure_manifest.csv',index=False)
    pd.DataFrame(concept_manifest).to_csv(OUTPUT_DIR/'concept_manifest.csv',index=False)
    metadata={'day':DAY,'seed':SEED,'N':N,'M':M,'dt_hours':dt,'gamma':gamma,'comfort_margin_c':comfort_margin_c,
        'true_initial_air_c':20.,'true_initial_fabric_c':20.,'solver':SOLVER,'solver_options':_solver_options,
        'platform':platform.platform(),'python':platform.python_version(),
        'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
        'runs':status_report.to_dict(orient='records'),'classroom_simulation_runtime_s':float(comparison.runtime_s.sum()),
        'cached_results':False,'data_provenance':'Supplied synthetic forecast table; named regeneration script absent; no fitted forecaster claimed.',
        'comparison_limits':'State feedback, forecast refresh and horizon endpoint change jointly.',
        'count_tolerance_c':1e-6,'objective_equivalence_absolute_tolerance_gbp':1e-6,
        'acceptance_absolute_tolerance':1e-6,'revision':'bracket-comment revision 2026-10-28','forecast_error_summary':forecast_error_summary.to_dict(orient='records')}
    (OUTPUT_DIR/'run_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')""")

    md('closing', r'''
    Task D completes the daily MPC run. Tutorial 8.2 uses this controller to
    examine the effects of different uncertainty sources.

    ## Optional activities

    Each activity requires the core tutorial and can be run independently of the
    other optional activities. Keep the same six workflow steps; change the
    objective, devices or horizon as specified below.

    The tracking and PV/battery loops stop on a rejected solve. Optional C
    retains the Task D backup policy. Supplied helpers check solutions, package logs and draw
    figures. Compare full-day costs only for completed runs.
    ''')

    md('optional-tracking', r'''
    ### Optional A. Track a temperature setpoint

    **Prerequisite:** the core tutorial only. Retain the same model, constraints
    and 48-interval horizon, and let $R^{\mathrm{end}}_\tau=(L^{\mathrm{end}}_\tau+U^{\mathrm{end}}_\tau)/2$ be the desired temperature.
    Penalise predicted deviations from that midpoint:

    $$\min\ \Delta t\sum_{\tau=k}^{k+N-1}\left[p_{\tau}P_{\tau|k}
      +\lambda_T(\widehat T_{\tau+1|k}-R^{\mathrm{end}}_\tau)^2+\gamma s_{\tau|k}\right].$$

    Choose $\lambda_T=1$ £/(°C² h) for this example. The problem is now
    quadratic. Setting $\lambda_T=0$ recovers the core cost objective.

    Both the fixed plan and MPC use this tracking objective, with separate
    buildings and matched noise. The table measures tracking using the realised
    temperature, although optimisation uses predicted temperature.

    **Inspect:** compare the two realised temperatures with the band midpoint.
    Does replanning reduce midpoint RMSE on this day? Report the bill and comfort
    deviations as well, using the realised temperatures to assess comfort.
    ''')

    code('optional-tracking-plan', r"""from importlib import reload
import scripts.week8_day1_transfer as transfer_helpers
import scripts.week8_day1_optional as optional_helpers
# Refresh edited helpers when reusing an existing notebook kernel.
reload(transfer_helpers)
reload(optional_helpers)
from scripts.week8_day1_transfer import solve_transfer
from scripts.week8_day1_optional import record_run, case_metrics, plot_pair, export_case
lambda_tracking=1.
tracking_runs={}""")

    md('optional-tracking-algorithm', r'''
    Use the same six steps for each policy, with a fresh building and matched
    noise. The fixed policy performs steps 2–3 once at $k=0$; MPC performs them at
    every interval. Both use the same quadratic tracking objective in step 3.

    ```text
    for policy in [fixed plan, MPC]:
        create a fresh building simulation
        for k = 0, ..., 47:
            1. Read measurement: read the stored indoor temperature.
            if policy is MPC or k is 0:
                2. Gather inputs: get the nowcast, forecasts and schedules.
                3. Optimise and check: solve the tracking problem and validate.
            4. Select action: move 0 for MPC; move k for the fixed plan.
            5. Apply and observe: apply once and store the next measurement.
            6. Repeat: use that measurement at the next origin.
    ```

    The code writes out the quadratic problem in step 3 for comparison with the
    formula above. Its horizon remains 48 intervals.
    ''')

    code('optional-tracking-run', r'''
    for policy_name in ['Fixed plan','MPC']:
        session=SimulationSession(operation,model,DAY,M=M,seed=SEED)
        saved_plans=[]
        horizon_rows=[]
        for k in range(M):
            # 1. Read measurement.
            view=session.available()
            if policy_name=='MPC' or k==0:
                # 2. Gather inputs.
                inputs=session.inputs(announced,forecasts,N)
                # 3. Optimise and check.
                power=cp.Variable(N)
                temperature=cp.Variable(N+1)
                slack=cp.Variable(N,nonneg=True)
                constraints=[temperature[0]==view['measurement_c'],
                    temperature[1:]==(1-b)*temperature[:-1]+b*inputs['outdoor']+c*COP*power,
                    power>=0,power<=P_MAX,temperature[1:]+slack>=inputs['lower'],temperature[1:]<=inputs['upper']]
                target=(inputs['lower']+inputs['upper'])/2
                objective=dt*(cp.sum(cp.multiply(inputs['price'],power)+gamma*slack)
                              +lambda_tracking*cp.sum_squares(temperature[1:]-target))
                problem=cp.Problem(cp.Minimize(objective),constraints)
                plan=solve_transfer(problem,{'power_kw':power,'temperature_c':temperature,'slack_c':slack},
                                    inputs,model,'tracking',gamma=gamma,lambda_tracking=lambda_tracking)
                if not plan['accepted']:
                    session.abort(plan['reason'])
                    break
                saved_plans.append(copy.deepcopy(plan))
            # 4. Select action.
            action_index=0 if policy_name=='MPC' else k
            action_kw=float(plan['power_kw'][action_index])
            # 5. Apply and observe.
            next_measurement=session.advance(action_kw,plan,prediction_index=action_index)
            horizon_rows.append({'interval':k,'origin':view['origin'],'N':len(plan['power_kw']),
                'endpoint':plan['valid_time'][-1]+pd.Timedelta(hours=dt)})
            # 6. Repeat.
        tracking_runs[policy_name]=record_run(session,saved_plans,horizon_rows)

    tracking_metrics=case_metrics(tracking_runs)
    display(tracking_metrics.round(3))
    tracking_figure=plot_pair(tracking_runs,'Optional A: temperature tracking')
    target_rows=tracking_runs['MPC']['intervals']
    tracking_figure.axes[0].plot(target_rows.interval_end,(target_rows.lower_c+target_rows.upper_c)/2,
                                 ':',color='black',label='Target midpoint')
    tracking_figure.axes[0].legend(loc='upper left',ncol=4)
    export_case(tracking_runs,tracking_metrics,tracking_figure,'optional_tracking',OUTPUT_DIR,FIGURE_DIR,EXPORT_ARTEFACTS)
    plt.show()
    plt.close(tracking_figure)
    ''')

    response('optional-tracking', 'Explain the tracking and cost trade-off.', 'MPC reduces realised midpoint RMSE from 0.769 to 0.220 degrees C, while the bill increases from GBP 6.178 to GBP 6.953. Both runs have zero sampled comfort-band violations on this day. Closer tracking within the band and lower electricity cost are distinct objectives.')

    md('optional-pv-text', r'''
    ### Optional B. Co-optimise heating, PV and a battery

    **Prerequisite:** the core tutorial. As in Tutorial 7.2, jointly choose
    heating, battery charging/discharging and grid import. Here available PV is
    uncertain and can be curtailed. The electricity balance covers heating, PV and
    storage; surplus PV is curtailed.

    For available PV forecast $\widehat q_{\tau|k}$, let $v_{\tau|k}$ be
    curtailed PV, $g_{\tau|k}$ grid import, and $P^{\mathrm{ch}}_{\tau|k},P^{\mathrm{dis}}_{\tau|k}$ the battery
    powers, all nonnegative. The balance is

    $$g_{\tau|k}+\widehat q_{\tau|k}-v_{\tau|k}+P^{\mathrm{dis}}_{\tau|k}
     =P_{\tau|k}+P^{\mathrm{ch}}_{\tau|k},\qquad 0\le v_{\tau|k}\le\widehat q_{\tau|k}.$$

    Stored energy evolves according to

    $$E_{\tau+1|k}=E_{\tau|k}+\Delta t\eta_cP^{\mathrm{ch}}_{\tau|k}
     -\Delta t P^{\mathrm{dis}}_{\tau|k}/\eta_d.$$

    Use the Week 7.2 settings: efficiencies 0.95, initial energy 5 kWh, energy
    bounds 0.5–9.5 kWh and charge/discharge limits of 3 kW. Both policies must end
    at midnight with 5 kWh. To compare bills with the same energy left over, this
    example plans over the **remaining day**, $N_k=48-k$. This uses a
    shrinking horizon from the first update. The core example retains its constant
    48-interval horizon.

    Minimise

    $$\Delta t\sum_{\tau=k}^{k+N_k-1}\left[p_{\tau}g_{\tau|k}+\gamma s_{\tau|k}
     +\epsilon(P^{\mathrm{ch}}_{\tau|k}+P^{\mathrm{dis}}_{\tau|k})\right],$$

    with $\epsilon=0.001$ £/kWh. The small throughput penalty discourages
    unnecessary cycling. The continuous formulation permits simultaneous charge
    and discharge; the results report any overlap. Keep the thermal dynamics and
    comfort constraints from the core problem. The battery state is modelled
    exactly here.

    We generate a separate noisy PV forecast at each origin by perturbing a
    synthetic daylight profile. Noise increases with lead time; forecasts are
    nonnegative and zero at night. The current PV reading is exact, while future
    PV forecasts have error. Both policies receive the same initial forecast. Only
    MPC uses subsequent forecast updates.
    ''')

    code('optional-pv-plan', r"""from importlib import reload
import scripts.week8_day1_transfer as transfer_helpers
import scripts.week8_day1_optional as optional_helpers
# Refresh edited helpers when reusing an existing notebook kernel.
reload(transfer_helpers)
reload(optional_helpers)
from scripts.week8_day1_transfer import solve_transfer
from scripts.week8_day1_optional import daylight_pv, record_run, battery_metrics, plot_pair, plot_battery, export_case
BATTERY={'eta_charge':.95,'eta_discharge':.95,'initial_kwh':5.,
         'lower_kwh':.5,'upper_kwh':9.5,'power_limit_kw':3.,'target_kwh':5.}
throughput_weight=.001

# Generate the synthetic experiment inputs once, before either controller runs.
pv_actual=pd.Series(daylight_pv(reporting_times),index=reporting_times)
pv_rng=np.random.default_rng(SEED+4101)  # separate from building/sensor noise
pv_forecast_frames=[]
for k,origin in enumerate(reporting_times):
    valid_time=reporting_times[k:]
    lead=np.arange(len(valid_time))
    reference_pv=pv_actual.loc[valid_time].to_numpy()
    std=.05+.25*np.sqrt(lead/48)
    predicted=np.clip(reference_pv*(1+pv_rng.normal(0,std)),0,2.)
    predicted[0]=reference_pv[0]  # current measurement; no future truth in the loop
    pv_forecast_frames.append(pd.DataFrame({'issue_time':origin,'lead_steps':lead,
        'valid_time':valid_time,'pv_forecast_kw':predicted}))
pv_forecasts=pd.concat(pv_forecast_frames,ignore_index=True)
future_pv=pv_forecasts.loc[pv_forecasts.lead_steps>0].copy()
future_pv['actual_pv_kw']=pv_actual.reindex(pd.DatetimeIndex(future_pv.valid_time)).to_numpy()
pv_forecast_rmse=float(np.sqrt(np.mean((future_pv.actual_pv_kw-future_pv.pv_forecast_kw)**2)))
print('Future PV forecast RMSE (kW):',round(pv_forecast_rmse,3))
export_table(pv_forecasts,'optional_pv_forecasts.csv')
pv_runs={}""")

    md('optional-pv-algorithm', r'''
    Extend the same workflow to include battery energy and the PV forecast.
    For MPC, plan over the remaining day. The fixed policy solves once at midnight
    and retains its initial commands.

    ```text
    for policy in [fixed plan, MPC]:
        create a fresh building; initialise battery energy to 5 kWh
        for k = 0, ..., 47:
            1. Read measurement: read indoor temperature and battery energy.
            if policy is MPC or k is 0:
                2. Gather inputs: get weather, schedules and PV to midnight.
                3. Optimise and check: solve the joint problem and validate.
            4. Select action: move 0 for MPC; move k for the fixed plan.
            5. Apply and observe: dispatch once and store the updated states.
            6. Repeat: use those states at the next origin.
    ```

    Step 4 selects heating, charging, discharging and a cap on PV output. The cap
    equals forecast availability minus planned curtailment. In step 5, actual PV
    output is limited by that cap, actual availability and demand; grid import
    supplies the remaining demand. Both policies use this dispatch rule. Calculate
    the bill from actual grid import and curtailed PV from availability minus
    delivered output.

    The optimisation uses the forecast issued at the current origin. Synthetic
    PV truth is used in step 5 to simulate operation, and battery energy carries
    forward from the previous dispatch.
    ''')

    code('optional-pv-run', r'''
    for policy_name in ['Fixed plan','MPC']:
        session=SimulationSession(operation,model,DAY,M=M,seed=SEED)
        battery_energy=BATTERY['initial_kwh']
        saved_plans=[]
        horizon_rows=[]
        for k in range(M):
            # 1. Read measurement.
            view=session.available()
            if policy_name=='MPC' or k==0:
                # 2. Gather inputs.
                n=M-k
                inputs=session.inputs(announced,forecasts,n)
                issued=pv_forecasts.loc[pv_forecasts.issue_time==view['origin']].sort_values('lead_steps')
                assert list(issued.valid_time)==list(inputs['valid_time'])
                pv=issued.pv_forecast_kw.to_numpy()
                # 3. Optimise and check.
                power=cp.Variable(n)
                temperature=cp.Variable(n+1)
                slack=cp.Variable(n,nonneg=True)
                grid=cp.Variable(n,nonneg=True)
                curtail=cp.Variable(n,nonneg=True)
                charge=cp.Variable(n,nonneg=True)
                discharge=cp.Variable(n,nonneg=True)
                energy=cp.Variable(n+1)
                constraints=[temperature[0]==view['measurement_c'],
                    temperature[1:]==(1-b)*temperature[:-1]+b*inputs['outdoor']+c*COP*power,
                    power>=0,power<=P_MAX,temperature[1:]+slack>=inputs['lower'],temperature[1:]<=inputs['upper'],
                    grid+pv-curtail+discharge==power+charge,curtail<=pv,
                    energy[0]==battery_energy,
                    energy[1:]==energy[:-1]+dt*BATTERY['eta_charge']*charge-dt*discharge/BATTERY['eta_discharge'],
                    energy>=BATTERY['lower_kwh'],energy<=BATTERY['upper_kwh'],
                    charge<=BATTERY['power_limit_kw'],discharge<=BATTERY['power_limit_kw'],
                    energy[-1]==BATTERY['target_kwh']]
                objective=dt*cp.sum(cp.multiply(inputs['price'],grid)+gamma*slack+throughput_weight*(charge+discharge))
                problem=cp.Problem(cp.Minimize(objective),constraints)
                variables={'power_kw':power,'temperature_c':temperature,'slack_c':slack,
                    'grid_import_kw':grid,'unused_pv_kw':curtail,'charge_kw':charge,'discharge_kw':discharge,'energy_kwh':energy}
                plan=solve_transfer(problem,variables,inputs,model,'pv_battery',pv=pv,gamma=gamma,
                    battery=BATTERY,energy_initial=battery_energy,throughput_weight=throughput_weight)
                if not plan['accepted']:
                    session.abort(plan['reason'])
                    break
                plan['pv_forecast_kw']=pv.copy()
                saved_plans.append(copy.deepcopy(plan))
            # 4. Select action.
            action_index=0 if policy_name=='MPC' else k
            heating=float(np.clip(plan['power_kw'][action_index],0,P_MAX))
            charging=float(np.clip(plan['charge_kw'][action_index],0,BATTERY['power_limit_kw']))
            discharging=float(np.clip(plan['discharge_kw'][action_index],0,BATTERY['power_limit_kw']))
            pv_output_cap=max(0.,float(plan['pv_forecast_kw'][action_index]-plan['unused_pv_kw'][action_index]))
            # 5. Apply and observe.
            available_pv=float(pv_actual.loc[view['origin']])
            net_demand=heating+charging-discharging
            assert net_demand>=-1e-6
            used_pv=min(pv_output_cap,available_pv,max(net_demand,0.))
            actual_import=max(net_demand-used_pv,0.)
            next_energy=battery_energy+dt*BATTERY['eta_charge']*charging-dt*discharging/BATTERY['eta_discharge']
            assert BATTERY['lower_kwh']-1e-6<=next_energy<=BATTERY['upper_kwh']+1e-6
            next_measurement=session.advance(heating,plan,prediction_index=action_index)
            session.intervals[-1].update(charge_kw=charging,discharge_kw=discharging,
                battery_start_kwh=battery_energy,battery_end_kwh=next_energy,
                requested_pv_output_kw=pv_output_cap,realised_pv_kw=available_pv,used_pv_kw=used_pv,
                curtailed_pv_kw=available_pv-used_pv,grid_import_kw=actual_import,
                planned_grid_import_kw=float(plan['grid_import_kw'][action_index]),
                interval_grid_cost_gbp=dt*session.intervals[-1]['price_gbp_per_kwh']*actual_import)
            battery_energy=next_energy
            horizon_rows.append({'interval':k,'origin':view['origin'],'N':len(plan['power_kw']),
                'endpoint':plan['valid_time'][-1]+pd.Timedelta(hours=dt)})
            # 6. Repeat.
        pv_runs[policy_name]=record_run(session,saved_plans,horizon_rows)

    pv_metrics=battery_metrics(pv_runs)
    display(pv_metrics.round(3).T)
    pv_figure=plot_pair(pv_runs,'Optional B: heating, uncertain PV and a battery',pv=True)
    export_case(pv_runs,pv_metrics,pv_figure,'optional_pv',OUTPUT_DIR,FIGURE_DIR,EXPORT_ARTEFACTS)
    plt.show()
    plt.close(pv_figure)
    ''')

    code('optional-pv-battery-figure', r"""battery_figure=plot_battery(pv_runs)
if EXPORT_ARTEFACTS:
    for ext in ['png','svg']:
        battery_figure.savefig(FIGURE_DIR/('optional_pv_battery_dispatch.'+ext),dpi=160)
plt.show();plt.close(battery_figure)""")

    response('optional-pv', 'Compare realised grid cost, curtailed PV and the final battery energy. Where does the battery shift energy?', textwrap.dedent(r'''
    Both policies start and finish with 5 kWh, giving equal stored energy at the
    horizon boundaries. Use the realised import and curtailment columns to
    evaluate operation. MPC updates the weather and PV forecasts and its measured
    thermal state; the battery follows the stated deterministic balance. Charging,
    discharging and PV curtailment must be interpreted jointly with heating and
    comfort. The observed bill difference applies to this noisy synthetic day.
    ''').strip())

    md('optional-shrinking-text', r'''
    ### Optional C. Compare rolling and shrinking horizons

    **Prerequisite:** Task D. Repeat its MPC experiment with the same day, seed,
    initial conditions, prediction/reference models, forecasts, prices, comfort
    bounds, heating limits, penalty and backup policy. Change the horizon rule:

    $$N_k=M-k=48-k,\qquad k=0,\ldots,47.$$

    Both controllers start with 48 intervals. Rolling MPC keeps $N=48$ and moves
    its endpoint forward. Shrinking MPC keeps the midnight endpoint and uses
    47 intervals at 00:30, 24 at 12:00 and one at 23:30. Both update every
    30 minutes and apply 48 actions over the day.

    The code marks two changes with `CHANGED`: choose `n=M-k`, then construct the
    same Task B problem with `n` controls. The vector lengths change, so the
    supplied `make_controller` helper recreates that formulation. The objective,
    constraints and failure handling retain their Task D settings.

    ```text
    Create a fresh building simulation with the Task D settings.
    for k = 0, ..., 47:
        1. Read measurement: read the stored indoor temperature.
        2. Gather inputs: use n = M - k; gather inputs to midnight.
        3. Optimise and check: solve the same problem with n controls.
        4. Select action: take the first move or use Task D backup handling.
        5. Apply and observe: apply once and store the next measurement.
        6. Repeat: use that measurement at the next origin.
    ```

    Compare the horizons, realised temperatures, heating and five operating
    metrics. Overlapping trajectories are a valid outcome; different horizons can
    select the same actions. Check that the first actions agree, every plan ends at
    midnight, and the last plan has one action and two temperature boundaries.
    The final predicted temperature uses the same lower-soft and upper-hard
    comfort requirements as Task D.
    ''')

    code('optional-shrinking-run', r'''
    from course_utils.week8_control import make_controller
    from importlib import reload
    import scripts.week8_day1_optional as optional_helpers
    # Refresh edited helpers when reusing an existing notebook kernel.
    reload(optional_helpers)
    from scripts.week8_day1_optional import record_run, export_case

    session=SimulationSession(operation,model,DAY,M=M,seed=SEED)
    policy=FallbackPolicy()
    last_command=None
    saved_plans=[]
    horizon_rows=[]
    for k in range(M):
        started=time.perf_counter()
        # 1. Read measurement.
        view=session.available()
        measured_temperature=view['measurement_c']
        # CHANGED 1: use the intervals remaining to midnight, in place of N=48.
        n=M-k
        if not np.isfinite(measured_temperature):
            plan=rejected_plan('Missing current temperature.',origin=view['origin'])
        else:
            try:
                # 2. Gather inputs.
                inputs=session.inputs(announced,forecasts,n)
                # 3. Optimise and check.
                # CHANGED 2: recreate the same Task B problem with n controls.
                local_controller=make_controller(
                    model,N=n,gamma=gamma,comfort_margin_c=comfort_margin_c)
                local_controller['initial'].value=measured_temperature
                for name in ['outdoor','price','lower','upper']:
                    local_controller[name].value=inputs[name]
                plan=solve_update(local_controller,inputs)
            except Exception as error:
                plan=rejected_plan(error,'invalid_inputs',view['origin'])
        # 4. Select action.
        if plan['accepted']:
            action_kw=first_action(plan)
            decision={'action_kw':action_kw,'fallback':False,'abort':False,'reason':''}
            policy.consecutive_missing=0
        else:
            decision=choose_command(plan,measured_temperature,view['lower_c'],view['upper_c'],P_MAX,policy,last_command)
        if decision['abort']:
            session.abort(decision['reason'])
            break
        if plan['accepted']:
            saved_plans.append(copy.deepcopy(plan))
        # 5. Apply and observe.
        next_measurement=session.advance(decision['action_kw'],plan,decision,time.perf_counter()-started)
        last_command=session.intervals[-1]['applied_power_kw']
        horizon_rows.append({'interval':k,'origin':view['origin'],'N':n,
            'endpoint':view['origin']+pd.Timedelta(hours=n*dt)})
        # 6. Repeat.
    shrinking_run=record_run(session,saved_plans,horizon_rows)
    shrinking_horizons=shrinking_run['horizons']
    display(shrinking_horizons.loc[shrinking_horizons.interval.isin([0,1,24,46,47])])

    metric_rows=[]
    for label,result in [('Rolling MPC',mpc_day),('Shrinking MPC',shrinking_run)]:
        score=realised_metrics(result)
        metric_rows.append({'Horizon':label,'Complete':score['complete'],
            'Energy (kWh)':score['energy_kwh'],'Bill (GBP)':score['bill_gbp'],
            'Peak (kW)':score['peak_kw'],'Below band (C h)':score['below_degree_hours'],
            'Above band (C h)':score['above_degree_hours']})
    shrinking_metrics=pd.DataFrame(metric_rows)
    display(shrinking_metrics.round(3))

    fig,axes=plt.subplots(3,1,figsize=(12.5,8),sharex=True)
    axes[0].stairs(np.full(M,N),mdates.date2num(state_time),baseline=None,
                   label='Rolling MPC',color='tab:blue',linestyle='--')
    axes[0].stairs(shrinking_horizons.N,mdates.date2num(shrinking_run['boundaries'].state_time),
                   baseline=None,label='Shrinking MPC',color='tab:orange')
    axes[0].set(ylabel='Planned intervals',yticks=[1,12,24,36,48],
                title='Same MPC setting: rolling N=48 versus shrinking N=48-k')
    for label,result,colour in [('Rolling MPC',mpc_day,'tab:blue'),
                                ('Shrinking MPC',shrinking_run,'tab:orange')]:
        rows=result['intervals']
        states=result['boundaries']
        line_style='--' if label=='Rolling MPC' else '-'
        line_order=3 if label=='Rolling MPC' else 2
        axes[1].plot(states.state_time,states.true_temperature_c,label=label,color=colour,
                      linestyle=line_style,zorder=line_order)
        axes[2].stairs(rows.applied_power_kw,mdates.date2num(states.state_time),
                       baseline=None,label=label,color=colour,linestyle=line_style,zorder=line_order)
    axes[1].plot(rows.interval_end,rows.lower_c,'--',color='0.4',label='Comfort bounds')
    axes[1].plot(rows.interval_end,rows.upper_c,'--',color='0.4')
    axes[1].set_ylabel('Indoor temperature (°C)')
    axes[2].set(ylabel='Applied power (kW)',xlabel='Time on 3 February 2025')
    for ax in axes:
        clock_axis(ax)
        ax.legend(loc='upper left',ncol=3)
    fig.tight_layout()
    export_case({'MPC':shrinking_run},shrinking_metrics,fig,'optional_shrinking',
                OUTPUT_DIR,FIGURE_DIR,EXPORT_ARTEFACTS)
    plt.show()
    plt.close(fig)
    ''')

    response('optional-shrinking','Compare rolling and shrinking horizons on the same day.', textwrap.dedent(r'''
    Both controllers start at midnight with 48 planned actions and the same
    initial information. Rolling MPC keeps 48 intervals at every update; shrinking
    MPC uses 48, 47, ..., 1, with a fixed midnight endpoint. Their first problems
    are equivalent. On this day, the realised actions and operating metrics agree
    within numerical precision: both bills round to GBP 3.459 and lower-band
    deviation to 0.652 degrees C h. Extra future intervals can leave the selected
    first move unchanged. Both controllers still apply 48 actions. The endpoint
    alone imposes no terminal-temperature equality. Other operating conditions
    can produce different current actions when later requirements affect the plan.
    ''').strip())

    code('optional-manifest', r"""if EXPORT_ARTEFACTS:
    optional_assets=[]
    for activity in ['tracking','pv','shrinking']:
        stem='optional_'+activity
        for ext,folder in [('png',FIGURE_DIR),('svg',FIGURE_DIR),('_metrics.csv',OUTPUT_DIR)]:
            file=folder/(stem+ext if ext.startswith('_') else stem+'.'+ext)
            optional_assets.append({'activity':activity,'file':str(file.relative_to(ROOT)),
                'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
                'producer_cell':'w81-optional-'+activity+'-run','optional':True})
    for name in ['optional_pv_battery_dispatch.png','optional_pv_battery_dispatch.svg']:
        file=FIGURE_DIR/name
        optional_assets.append({'activity':'pv','file':str(file.relative_to(ROOT)),
            'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
            'producer_cell':'w81-optional-pv-battery-figure','optional':True})
    pd.DataFrame(optional_assets).to_csv(OUTPUT_DIR/'optional_activity_manifest.csv',index=False)""")

    md('references', r"""## References and source credits (slide 37)

- [Reference-building README](data/reference_building/README.md): supplied synthetic model and data.
- Week 3, PDF pp. 49–55 and 105: BAS/EMIS context and residual notation, as mapped in the supplied revision brief.
- Tutorials 6.1–6.2 and 7.1: thermal dynamics, electrical power, endpoint comfort and midpoint thermostat.
- Powell, W. B. (2019), [*A unified framework for stochastic optimization*](https://warrenpowell.org/assets/papers/2019/02/Powell-A-Unified-Framework-for-Stochastic-Optimization-EJOR.pdf), European Journal of Operational Research 275, 795–821, Sections 4 and 9.4: information timing and the distinction between operating and lookahead models. Here we retain the course hat convention and write target time before the bar, with planning time after it.
- Drgoňa, J. et al. (2020), *All you need to know about model predictive control for buildings*, Annual Reviews in Control 50, 190–232, DOI: 10.1016/j.arcontrol.2020.09.001. C81-01/C81-03/C81-04 are original course redraws informed by Figs. 2–3 and Algorithm 1, with simplified course notation; schematic curves are not empirical results.
- Diamond, S. and Boyd, S. (2016), *CVXPY: A Python-Embedded Modeling Language for Convex Optimization*, JMLR 17(83), 1–5.
- [do-mpc: basics of MPC](https://www.do-mpc.com/en/latest/theory_mpc.html): repeated optimisation with a shifted prediction horizon.
- [EnergyPlus project](https://energyplus.net/): example of a detailed building-energy simulator; not used to generate this tutorial's results.""")

    md('reference-appendix',r'''
    ## Appendix A. Full reference model and parameter values

    The reference building has two thermal states: indoor air temperature $T_a$
    and fabric temperature $T_f$. The schematic shows their heat-exchange paths,
    thermal storage and heat inputs. The sensor reports air temperature with
    measurement noise.

    The next cell saves the schematic as SVG, PNG and PDF in
    `figures/week8_day1/reference_model_schematic.*`. The SVG retains editable
    text and vector shapes for use in slides.
    ''')

    code('reference-schematic',r'''
    from importlib import reload
    import scripts.week8_day1_visuals as reference_visuals
    reload(reference_visuals)

    reference_figure=reference_visuals.reference_schematic()
    if EXPORT_ARTEFACTS:
        FIGURE_DIR.mkdir(parents=True,exist_ok=True)
        for extension in ['svg','png','pdf']:
            reference_figure.savefig(
                FIGURE_DIR/f'reference_model_schematic.{extension}',dpi=240)
    plt.show()
    plt.close(reference_figure)
    ''')

    md('reference-appendix-details',r'''
    With time measured in hours, the deterministic balances are

    $$C_a\dot T_a=\frac{T_f-T_a}{R_{af}}+G_v(T_o-T_a)
     +\alpha_g Q_g+\alpha_h Q_{hp},$$
    $$C_f\dot T_f=\frac{T_a-T_f}{R_{af}}+\frac{T_o-T_f}{R_{of}}
     +(1-\alpha_g)Q_g+(1-\alpha_h)Q_{hp}.$$

    $$Q_g=A_s\max(I,0)+\max(Q_{int},0),\qquad
    Q_{hp}=\mathrm{COP}_{ref}(T_o)P^{act},$$
    $$\mathrm{COP}_{ref}(T_o)=\operatorname{clip}
    \big(3+0.05(T_o-5),\,2,\,4.2\big),\qquad
    P^{act}=P_{max}\operatorname{clip}(u,0,1).$$

    | Parameter | Value | Unit / role |
    |---|---:|---|
    | $R_{af}$ | 0.8 | K/kW, air–fabric exchange |
    | $R_{of}$ | 6.0 | K/kW, outdoor–fabric exchange |
    | $C_a$ | 0.8 | kWh/K, air capacitance |
    | $C_f$ | 18.0 | kWh/K, fabric capacitance |
    | $G_v$ | 0.08 | kW/K, ventilation conductance |
    | $A_s$ | 0.010 | kW/(W/m²), solar-gain coefficient |
    | $\alpha_g$ | 0.45 | Fraction of gains applied to air |
    | $\alpha_h$ | 0.25 | Fraction of heat applied to air |
    | $P_{max}$ | 5.0 | Electrical kW |
    | COP at 5°C / slope | 3.0 / 0.05 | Ratio / K⁻¹ |
    | Minimum / maximum COP | 2.0 / 4.2 | Ratio |
    | Initial $T_a,T_f$ | 20.0, 20.0 | °C |
    | Reporting interval / internal substeps | 0.5 / 6 | h / count |
    | Air process-noise standard deviation | 0.01 | °C per reporting interval |
    | Fabric process-noise standard deviation | $0.01/3$ | °C per reporting interval |
    | Sensor-noise standard deviation | 0.05 | °C per observation |
    | Main-day seed | 88 | Reproducible simulation |

    At each reporting interval the code holds $P^{act},T_o,I,Q_{int}$ constant,
    performs six explicit Euler substeps of length $h=0.5/6$ h, then adds independent
    process perturbations $w_{a,k+1}\sim\mathcal N(0,0.01^2)$ and
    $w_{f,k+1}\sim\mathcal N(0,(0.01/3)^2)$ at the next boundary. The observation is
    $T^{meas}_{k+1}=T_{a,k+1}+\nu_{k+1}$, with
    $\nu\sim\mathcal N(0,0.05^2)$. There is one initial observation and one returned
    observation per transition. The main case has no extra sensor bias.

    The notebook selects $P^*_{k|k}$ and calls `session.advance(power_kw, plan)`.
    The supplied adapter converts kW to modulation $u=P/P_{max}$ and calls
    `TwoStateReferencePlant.step(u, realised_outdoor, realised_solar, realised_gains)`.
    The reference evolves both states and returns the next noisy indoor reading.
    That reading initialises the next prediction; neither physical state is reset
    by MPC. The fixed plan and thermostat use the same actuator/reference interface.

    These reference parameters are selected teaching values, as documented in the
    data README. They are inspired by BOPTEST but are not an extracted or validated
    BOPTEST reduction. The CSVs are supplied synthetic data and the named original
    generation script is absent. Thus “true” means the underlying simulated state,
    not an empirically established real-building truth. Model mismatch, imperfect
    future forecasts and sensor/process noise coexist here. Tutorial 8.2 introduces
    a matched one-state reference to vary one error channel at a time, then returns
    to this richer reference as a combined-error transfer example.
    ''')
    return nbf.v4.new_notebook(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python', 'version': '3'}})

def write_metadata():
    teaching = ROOT / 'teaching'
    teaching.mkdir(exist_ok=True)
    with (teaching / 'week8_day1_task_manifest.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task_id', 'slide', 'minutes_including_debrief', 'student_work_minutes', 'debrief_minutes', 'cell_id'])
        writer.writeheader()
        for letter, slide, minutes in TASKS:
            debrief = 4 if letter in 'DE' else 3
            writer.writerow(dict(task_id=letter, slide=slide, minutes_including_debrief=minutes, student_work_minutes=minutes - debrief, debrief_minutes=debrief, cell_id='w81-task-' + letter.lower()))
    with (teaching / 'week8_day1_schedule.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['slide', 'block', 'core_minutes', 'task', 'role', 'break_after_minutes', 'guided_minutes'])
        writer.writeheader()
        for slide in range(1, 41):
            task = next((letter for letter, s, _ in TASKS if s == slide), '')
            writer.writerow(dict(slide=slide, block=(slide - 1) // 9 + 1 if slide <= 36 else 0, core_minutes=SLIDE_MINUTES[slide - 1] if slide <= 36 else 0, task=task, role='Task and debrief' if task else 'Core' if slide <= 36 else 'References' if slide == 37 else 'Optional', break_after_minutes='15–20 additional' if slide == 18 else '', guided_minutes={5:4,9:6,17:5,23:7,25:4,34:5,35:12}.get(slide,0)))
if __name__ == '__main__':
    for solution in [False, True]:
        nbf.write(make_notebook(solution), ROOT / (STEM + ('_solution' if solution else '') + '.ipynb'))
    write_metadata()
    print('Built Tutorial 8.1: six tasks; 71 task minutes within 200 teaching minutes.')
