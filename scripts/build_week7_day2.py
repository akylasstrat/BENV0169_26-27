"""Build the independent Tutorial 7.2 student/solution pair and teaching metadata."""
from pathlib import Path
import csv
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STEM = 'tutorial_7_2_flexible_building_assets'
TASKS = [('A',6,12),('B',9,10),('C',14,12),('D',17,12),('E',21,10),('F',26,14)]
SLIDE_MINUTES = [2,3,5,5,5,12,10,8,10,5,5,4,4,12,6,4,12,5,4,5,10,4,5,5,5,14,6,6,5,5,5,4]
FIGURES = [('F72-01', 4, 'w7d2_F01_site_inputs', 'Supplied first-day weather, endpoint comfort, base demand, PV and tariffs; known planning inputs.', 'A'), ('F72-02', 8, 'w7d2_F02_site_balance', 'Symbolic site balance: import, PV and discharge supply base demand, heating, charging and export.', 'B'), ('F72-03', 12, 'w7d2_F03_battery_time_index', 'Four interval powers connect five stored-energy boundaries; full case uses 48 intervals and 49 boundaries.', 'C'), ('F72-04', 17, 'w7d2_F04_site_operation', 'Joint cost-only case C: heating, base demand, PV, battery powers, grid exchanges and import tariff.', 'D'), ('F72-05', 18, 'w7d2_F05_stored_energy', 'Case C stored energy at all 49 marked boundaries, with straight connecting segments, bounds and equal initial/final 5 kWh.', 'D'), ('F72-06', 19, 'w7d2_F06_operating_summary', 'Case C import cost, export revenue, net bill, charging/discharging energy, battery losses and initial/final energy.', 'E'), ('F72-07', 21, 'w7d2_F07_terminal_comparison', 'Fixed versus unrestricted final battery energy: compare the bill with energy left for the next day.', 'E'), ('F72-08', 29, 'w7d2_F08_cost_peak_comparison', 'Optional same-asset C comparison: cost-only versus bill plus 0.35 GBP/kW times peak import.', 'Optional'), ('F72-09', 29, 'w7d2_F09_event_flexibility', 'Optional same-asset objective comparison: signed baseline-minus-candidate import energy before, during and after 16:00–19:00.', 'Optional'), ('F72-10', 25, 'w7d2_F10_evidence_table', 'Core A/B/C bills, import/export energy, actual peak and both comfort deviations under matched conditions.', 'F'), ('F72-11', 11, 'w7d2_F11_pv_comparison', 'A versus B: re-optimised heating, import/export and endpoint comfort under the same tariff and weather.', 'B'), ('F72-12', 24, 'w7d2_F12_asset_comparison', 'A/B/C grid imports with PV/tariff context; bill, imported energy and actual peak shown separately.', 'F')]


def make_notebook(solution=False):
    cells=[]
    def md(key,source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(),id='w72-'+key))
    def code(key,source,tags=None):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(),id='w72-'+key,
                                        metadata={'tags':tags or ['supplied']}))
    def task(letter,answer,scaffold):
        code('task-'+letter.lower(),answer if solution else scaffold,['task',letter])
    def interpretation(letter,prompt,answer):
        md('response-'+letter.lower(),f'**Written interpretation {letter}.** {prompt}\n\n'+
           (answer if solution else '*Write your interpretation here.*'))

    md('purpose',r'''
    # Tutorial 7.2: Dynamic optimisation and flexible building assets

    Extend the hard-constrained heating problem with PV generation and battery
    storage. Heating remains a decision in every configuration. We compare net
    site bills under common weather, tariffs, thermal requirements and initial
    conditions.

    **Outcomes:** formulate the site balance and battery dynamics; assemble a
    linear program; interpret operating schedules; distinguish the operating value
    of PV from that of storage; interpret bill, energy, peak and comfort
    separately.

    | Block | Slides | Minutes | Integrated work |
    |---|---|---:|---|
    | 1 | 1–8 | 50 | Site formulation and heating-only baseline; Task A |
    | 2 | 9–16 | 50 | PV and battery equations; Tasks B–C |
    | Break | after 16 | 15–20 additional | Save and pause |
    | 3 | 17–22 | 50 | Joint operation, electricity costs and battery energy; Tasks D–E |
    | 4 | 23–26 | 50 | Matched asset comparison; Task F |

    Task times include debrief: A 12, B 10, C 12, D 12, E 10 and F 14 minutes (70
    within 200 teaching minutes). References are on slide 27; slides 28–30 are
    optional. Run this notebook from the repository root. It loads the inputs
    needed to run independently of Tutorial 7.1.
    ''')
    code('setup',r'''
    from pathlib import Path
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import cvxpy as cp
    from IPython.display import display

    ROOT=Path.cwd()
    DATA_FILE=ROOT/'data/reference_building/operation_conditions.csv'
    MODEL_FILE=ROOT/'data/reference_building/verified_controller_model.json'
    if not DATA_FILE.is_file() or not MODEL_FILE.is_file():
        raise FileNotFoundError('Run from the repository root containing data/reference_building.')
    available_solvers=cp.installed_solvers()
    SOLVER=next((s for s in ['CLARABEL','SCIPY','OSQP','SCS'] if s in available_solvers),None)
    if SOLVER is None:raise RuntimeError('An installed open-source CVXPY solver is required.')
    TRUE_T0=20.0
    ETA_CHARGE=ETA_DISCHARGE=0.95
    E_INIT=5.0; E_MIN=0.5; E_MAX=9.5; BATTERY_MAX=3.0
    EXPORT_ARTEFACTS=True
    FIGURE_DIR=ROOT/'figures/week7_day2'
    OUTPUT_DIR=ROOT/'teaching/week7_day2_outputs'
    plt.rcParams.update({'figure.dpi':110,'font.size':12,'axes.grid':True,
        'grid.alpha':.2,'svg.fonttype':'none','savefig.facecolor':'white'})
    COLOURS={'A':'tab:blue','B':'tab:green','C':'tab:orange','peak':'tab:purple',
             'pv':'goldenrod','base':'0.4','charge':'tab:blue','discharge':'tab:red'}
    CASE_NAMES={'A':'A: Heating only','B':'B: Heating + PV','C':'C: Heating + PV + battery'}
    print('CVXPY',cp.__version__,'selected solver:',SOLVER)
    ''')
    md('inputs-text',r'''
    ## 1. Common planning case (slides 3–5)

    The first 48 rows of the supplied synthetic case describe one day. Each
    timestamp marks an interval start. The discrete RC coefficients and fixed COP
    match Tutorial 7.1. Base demand excludes the heat pump; available PV is a
    fixed injection. Grid exports accommodate surplus generation.

    Future weather, demand, PV and tariffs are known planning inputs. Results are
    conditional on these supplied profiles. The [data
    README](data/reference_building/README.md) describes the synthetic reference
    plant and its limitations. Use the supplied profiles and operating day
    throughout the comparisons.

    The indoor air-temperature band is a simplified operating requirement. Week 5
    considered additional environmental and personal factors in comfort.
    ''')
    code('inputs',r'''
    conditions=pd.read_csv(DATA_FILE,index_col='timestamp',parse_dates=True).iloc[:48].copy()
    model=json.loads(MODEL_FILE.read_text(encoding='utf8'))
    dt=float(model['sampling_interval_hours'])
    b=float(model['loss_coefficient_per_step'])
    c=float(model['heat_coefficient_c_per_kw_step'])
    COP=float(model['nominal_cop'])
    P_MAX=float(model['heat_pump_max_electric_kw'])
    N=len(conditions)
    required=['outdoor_temperature_c','comfort_min_c','comfort_max_c','base_load_kw',
              'pv_available_kw','import_price_gbp_per_kwh','export_price_gbp_per_kwh']
    expected=pd.date_range(conditions.index[0],periods=48,freq='30min')
    if N!=48 or dt!=.5 or not conditions.index.equals(expected):
        raise ValueError('Expected 48 contiguous half-hour intervals and dt=0.5 h.')
    if not np.isfinite(conditions[required]).all().all():raise ValueError('Inputs must be finite.')
    if not np.isfinite([b,c,COP,P_MAX]).all() or not (0<b<1 and min(c,COP,P_MAX)>0):
        raise ValueError('Inspect model coefficients and units.')
    outdoor=conditions.outdoor_temperature_c.to_numpy(copy=True)
    lower=conditions.comfort_min_c.to_numpy(copy=True)
    upper=conditions.comfort_max_c.to_numpy(copy=True)
    base_load=conditions.base_load_kw.to_numpy(copy=True)
    pv_available=conditions.pv_available_kw.to_numpy(copy=True)
    import_prices=conditions.import_price_gbp_per_kwh.to_numpy(copy=True)
    export_prices=conditions.export_price_gbp_per_kwh.to_numpy(copy=True)
    if np.any(lower>upper) or np.any(base_load<0) or np.any(pv_available<0):
        raise ValueError('Invalid comfort bounds or negative supplied power.')
    if not ((import_prices>0)&(import_prices>export_prices)&(export_prices>=0)).all():
        raise ValueError('This case requires positive import prices above nonnegative export prices.')
    interval_start=pd.DatetimeIndex(conditions.index,name='interval_start')
    interval_end=interval_start+pd.Timedelta(hours=dt)
    state_time=pd.DatetimeIndex([interval_start[0],*interval_end],name='state_time')
    display(pd.Series({'start':interval_start[0],'end':state_time[-1],'intervals':N,
        'boundaries':len(state_time),'dt_h':dt,'initial_temperature_c':TRUE_T0,
        'b':b,'c_C_per_kW_step':c,'COP':COP,'P_MAX_kW':P_MAX},name='Common case'))
    ''')
    code('plot-helpers','FIGURE_SPECS='+repr(FIGURES)+'\n'+textwrap.dedent(r'''
    figure_manifest=[]
    source_paths=['data/reference_building/operation_conditions.csv',
                  'data/reference_building/verified_controller_model.json',
                  'scripts/build_week7_day2.py','scripts/week7_day2_support.py']

    def export_table(frame,name,index=False):
        if EXPORT_ARTEFACTS:
            OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
            frame.to_csv(OUTPUT_DIR/name,index=index)

    def save_figure(fig,figure_id,case,counts='48 intervals; 49 boundaries'):
        spec=next(row for row in FIGURE_SPECS if row[0]==figure_id)
        symbolic=figure_id in {'F72-02','F72-03'}
        fig.tight_layout()
        if EXPORT_ARTEFACTS:
            FIGURE_DIR.mkdir(parents=True,exist_ok=True)
            for ext in ['png','svg']:fig.savefig(FIGURE_DIR/f'{spec[2]}.{ext}',dpi=220)
            figure_manifest.append({'id':figure_id,'slide':spec[1],'task':spec[4],
                'use':'Optional' if spec[1]>32 else 'Core','case':case,
                'png':f'figures/week7_day2/{spec[2]}.png','svg':f'figures/week7_day2/{spec[2]}.svg',
                'notebook':'tutorial_7_2_flexible_building_assets_solution.ipynb',
                'placement':('12.5 x 7 inch frame; preserve aspect ratio' if figure_id=='F72-04' else
                             '12.5 x 3.5 inch frame; preserve aspect ratio'),'caption':spec[3],
                'period':'Symbolic' if symbolic else f'{interval_start[0]} to {state_time[-1]}',
                'dt_hours':dt,'N_intervals':4 if figure_id=='F72-03' else (0 if symbolic else N),
                'counts':counts,'units':'kW;kWh;h;degrees C;GBP;GBP/kWh',
                'assumptions':'Symbolic diagram, not measured flows' if symbolic else
                    'Known inputs; fixed COP; hard endpoint comfort; final thermal comfort only; fixed PV; export allowed; battery terminal equality except named end-of-day experiment',
                'source_files':';'.join(source_paths),
                'source_sha256':';'.join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths)})
        plt.show();plt.close(fig)

    def clock_axis(ax):
        ax.set_xlim(state_time[0],state_time[-1])
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,4)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    def comfort_lines(ax):
        ax.plot(state_time[1:],lower,'.--',color='black',markersize=3,label='Endpoint bounds')
        ax.plot(state_time[1:],upper,'.--',color='black',markersize=3)

    def draw_table(ax,frame,fontsize=11,scale=1.7,widths=None):
        shown=frame.copy()
        for col in shown.select_dtypes(include='floating'):
            # Display rounding only; exports retain unaltered arrays.
            shown[col]=shown[col].map(lambda x:'n/a' if pd.isna(x) else f'{0. if round(x,3)==0 else x:.3f}')
        ax.axis('off')
        table=ax.table(cellText=shown.values,colLabels=shown.columns,loc='center',
                       cellLoc='center',colWidths=widths)
        table.auto_set_font_size(False);table.set_fontsize(fontsize);table.scale(1,scale)
        for (row,col),cell in table.get_celld().items():
            cell.set_edgecolor('0.8')
            if row==0:cell.set_facecolor('0.92')
        return table

    def export_schedule(result,stem):
        require_checked(result)
        frame=pd.DataFrame({'interval_start':interval_start,'interval_end':interval_end,
            **{key:result[key] for key in ['power_kw','grid_import_kw','grid_export_kw',
                                         'charge_kw','discharge_kw','pv_kw']}})
        export_table(frame,stem+'_intervals.csv')
        states=pd.DataFrame({'state_time':state_time,'temperature_c':result['temperature_c']})
        if result['battery']:states['stored_energy_kwh']=result['energy_kwh']
        export_table(states,stem+'_states.csv')
    ''').strip())
    code('input-figure',r'''
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(12.5,3.5))
    axes[0].stairs(outdoor,state_time,label='Outdoor',color='tab:blue')
    comfort_lines(axes[0]);axes[0].set_ylabel('T (°C)')
    axes[1].stairs(base_load,state_time,label='Base demand',color=COLOURS['base'])
    axes[1].stairs(pv_available,state_time,label='Available PV',color=COLOURS['pv'])
    axes[1].set_ylabel('Power (kW)')
    axes[2].stairs(import_prices,state_time,label='Import price',color='black')
    axes[2].stairs(export_prices,state_time,label='Export price',color='tab:green')
    axes[2].set(ylabel='£/kWh',xlabel='Interval time')
    for ax in axes:
        clock_axis(ax);ax.legend(loc='center left',bbox_to_anchor=(1.01,.5),fontsize=10)
    save_figure(fig,'F72-01','Supplied first day')
    export_table(conditions.reset_index(),'planning_inputs.csv')
    ''')
    md('heating-text',r'''
    ## 2. Retain the hard-comfort heating model

    Electrical heating power $P_k$ is in kW and temperature $T_j$ is in °C.
    For $k=0,\ldots,N-1$, the supplied model is
    $$T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k.$$
    We retain $T_0=20$°C, $0\le P_k\le P_{\max}$ and
    $L_k\le T_{k+1}\le U_k$. The coefficient $c$ already includes the time step; an additional
    $\Delta t$ would apply that scaling twice. The model uses the supplied thermal terms, hard comfort bounds and a
    bill objective. The last temperature is free within its comfort band,
    as in Tutorial 7.1.

    The cells below supply these already-taught constraints. CVXPY Parameters
    store the known profiles, while Variables represent the decisions.
    ''')
    code('thermal-formulation',r'''
    outdoor_parameter=cp.Parameter(N,value=outdoor,name='outdoor_c')
    lower_parameter=cp.Parameter(N,value=lower,name='lower_c')
    upper_parameter=cp.Parameter(N,value=upper,name='upper_c')
    base_parameter=cp.Parameter(N,nonneg=True,value=base_load,name='base_kw')
    pv_parameter=cp.Parameter(N,nonneg=True,value=pv_available,name='available_pv_kw')
    import_price=cp.Parameter(N,nonneg=True,value=import_prices,name='import_GBP_per_kWh')
    export_price=cp.Parameter(N,nonneg=True,value=export_prices,name='export_GBP_per_kWh')
    P=cp.Variable(N,name='heating_electric_kw')
    T=cp.Variable(N+1,name='temperature_c')
    heating_constraints=[T[0]==TRUE_T0,
        T[1:]==(1-b)*T[:-1]+b*outdoor_parameter+c*COP*P,
        P>=0,P<=P_MAX,T[1:]>=lower_parameter,T[1:]<=upper_parameter]
    ''')
    md('solve-text',r'''
    Run the supplied solve-helper cell once. It returns checked schedules for the
    comparisons and stops with a message if a solve fails the checks. Your tasks
    focus on the formulation and operating results.
    ''')
    code('solve-helper',r'''
    from scripts.week7_day2_support import SiteChecks, require_checked

    case_checks=SiteChecks(outdoor=outdoor,lower=lower,upper=upper,base_load=base_load,
        import_prices=import_prices,export_prices=export_prices,
        dt=dt,b=b,c=c,COP=COP,P_MAX=P_MAX,TRUE_T0=TRUE_T0,
        ETA_CHARGE=ETA_CHARGE,ETA_DISCHARGE=ETA_DISCHARGE,
        E_INIT=E_INIT,E_MIN=E_MIN,E_MAX=E_MAX,BATTERY_MAX=BATTERY_MAX,solver=SOLVER)
    solve_checked=case_checks.solve_checked
    ''')
    md('case-a-formulation',r'''
    ### Case A: complete heating-only formulation

    Choose heating powers $P_k$, temperatures $T_j$ and grid powers
    $P_k^{\mathrm{imp}},P_k^{\mathrm{exp}}$. The full site problem for Task A is
    $$
    \begin{aligned}
    \min_{P,T,P^{\mathrm{imp}},P^{\mathrm{exp}}}\quad
        &\Delta t\sum_{k=0}^{N-1}
        \left(p_k^{\mathrm{imp}}P_k^{\mathrm{imp}}
        -p_k^{\mathrm{exp}}P_k^{\mathrm{exp}}\right)\\
    \text{subject to}\quad
        &T_0=T_{\mathrm{init}}=20\ ^\circ\mathrm{C},\\
        &T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k,\\
        &0\le P_k\le P_{\max},\qquad L_k\le T_{k+1}\le U_k,\\
        &P_k^{\mathrm{imp}}=P_k^{\mathrm{base}}+P_k,\qquad
          P_k^{\mathrm{exp}}=0,\\
        &P_k^{\mathrm{imp}},P_k^{\mathrm{exp}}\ge0,
        \qquad k=0,\ldots,N-1.
    \end{aligned}
    $$
    Temperature has $N+1$ entries, indexed by $j=0,\ldots,N$. PV, charge and
    discharge are identically zero; battery energy states are absent. The
    objective equals the heating bill plus the constant base-demand bill.

    A fixed term in an objective changes its value but not its optimal decisions.
    The base-demand bill is fixed here because base load and prices are inputs.
    ''')
    code('baseline',r'''
    # Supplied case A: no PV or battery. These grid variables retain the site formulation.
    import_A=cp.Variable(N,nonneg=True,name='import_A_kw')
    export_A=cp.Variable(N,nonneg=True,name='export_A_kw')
    constraints_A=heating_constraints+[import_A==base_parameter+P,export_A==0]
    bill_A=dt*cp.sum(cp.multiply(import_price,import_A)-cp.multiply(export_price,export_A))
    problem_A=cp.Problem(cp.Minimize(bill_A),constraints_A)
    case_A=require_checked(solve_checked(problem_A,{'power_kw':P,'temperature_c':T,
        'grid_import_kw':import_A,'grid_export_kw':export_A,
        'charge_kw':np.zeros(N),'discharge_kw':np.zeros(N)},pv=np.zeros(N),case=CASE_NAMES['A']))
    ''')
    md('task-a-text',r'''
    ### Task A — recover the heating case (slide 6; 12 min including debrief)

    Classify heating, temperature, tariffs, base demand and PV as decisions or
    parameters. From the returned arrays calculate import energy, the site bill,
    heat-pump electricity and both comfort degree-hours. Explain why adding the
    fixed base-load bill preserves the set of optimal heating schedules.
    ''')
    task('A',r'''
    baseline_metrics={
        'bill_gbp':float(dt*np.sum(import_prices*case_A['grid_import_kw']-export_prices*case_A['grid_export_kw'])),
        'import_kwh':float(dt*np.sum(case_A['grid_import_kw'])),
        'heat_pump_kwh':float(dt*np.sum(case_A['power_kw'])),
        'below_band_degree_hours':float(dt*np.maximum(lower-case_A['temperature_c'][1:],0.).sum()),
        'above_band_degree_hours':float(dt*np.maximum(case_A['temperature_c'][1:]-upper,0.).sum())}
    display(pd.Series(baseline_metrics,name=CASE_NAMES['A']))
    ''',r'''
    # TODO A: compute baseline_metrics from case_A arrays and display the result.
    # Include bill_gbp, import_kwh, heat_pump_kwh and both comfort degree-hours.
    raise NotImplementedError('Task A: calculate the heating-only baseline metrics.')
    ''')
    interpretation('A','Which quantities are decisions, and why does\nadding base demand preserve the heating optimum?',
        textwrap.dedent(r'''
    Heating power and temperature are decisions. Weather, comfort bounds, tariffs,
    base demand and available PV are parameters. In A, import equals base demand
    plus heating, and export is zero. The fixed base-demand bill adds the same
    amount to every heating schedule. Minimising the site bill therefore preserves
    the optimal heating schedules. Several schedules may achieve the same optimal
    bill.
    ''').strip())
    md('balance-text',r'''
    ## 3. Add PV through the site balance (slides 7–10)

    Import, export, charge and discharge are nonnegative powers. Our sign convention is
    $$P_k^{\mathrm{imp}}+P_k^{\mathrm{PV}}+P_k^{\mathrm{dis}}
      =P_k^{\mathrm{base}}+P_k+P_k^{\mathrm{ch}}+P_k^{\mathrm{exp}}.$$
    In case B, both battery powers are zero and the model uses heating
    and grid variables.
    The net bill is
    $$J_{\mathrm{bill}}=\Delta t\sum_k
      (p_k^{\mathrm{imp}}P_k^{\mathrm{imp}}-p_k^{\mathrm{exp}}P_k^{\mathrm{exp}}).$$
    Heating purchases are included in imports; adding a second heating bill would
    double-count them. PV used locally forgoes export revenue, so its opportunity
    cost depends on the tariff. Heating is re-optimised with the original PV profile.

    The supplied import tariff is strictly greater than the nonnegative export
    tariff. With unlimited import and export, an export tariff above the import
    tariff would let both flows increase together and make the cost objective
    unbounded below. Nonnegative variables permit simultaneous import and export.
    The supplied checks inspect the returned flows.
    ''')
    code('balance-diagram',r'''
    fig,ax=plt.subplots(figsize=(12.5,3.5));ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    ax.plot([.5,.5],[.08,.92],color='black',linewidth=5)
    for y,label in zip([.8,.5,.2],['Grid import $P_k^{imp}$','Available PV $P_k^{PV}$','Discharge $P_k^{dis}$']):
        ax.annotate('',xy=(.49,y),xytext=(.29,y),arrowprops={'arrowstyle':'->','lw':2})
        ax.text(.27,y,label,ha='right',va='center',fontsize=14)
    for y,label in zip([.85,.62,.38,.15],['Base demand $P_k^{base}$','Heating $P_k$','Charge $P_k^{ch}$','Grid export $P_k^{exp}$']):
        ax.annotate('',xy=(.71,y),xytext=(.51,y),arrowprops={'arrowstyle':'->','lw':2})
        ax.text(.73,y,label,ha='left',va='center',fontsize=14)
    ax.set_title('Site electricity balance: all arrows denote nonnegative powers (kW)')
    save_figure(fig,'F72-02','Symbolic site bus','3 supplies; 4 uses; not measured flows')
    ''')
    md('case-b-formulation',r'''
    ### Case B: complete heating-plus-PV formulation

    Retain the same decisions and thermal requirements. The fixed available PV
    profile enters the site balance, giving the full problem for Task B:
    $$
    \begin{aligned}
    \min_{P,T,P^{\mathrm{imp}},P^{\mathrm{exp}}}\quad
        &\Delta t\sum_{k=0}^{N-1}
        \left(p_k^{\mathrm{imp}}P_k^{\mathrm{imp}}
        -p_k^{\mathrm{exp}}P_k^{\mathrm{exp}}\right)\\
    \text{subject to}\quad
        &T_0=T_{\mathrm{init}}=20\ ^\circ\mathrm{C},\\
        &T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k,\\
        &0\le P_k\le P_{\max},\qquad L_k\le T_{k+1}\le U_k,\\
        &P_k^{\mathrm{imp}}+P_k^{\mathrm{PV}}
          =P_k^{\mathrm{base}}+P_k+P_k^{\mathrm{exp}},\\
        &P_k^{\mathrm{imp}},P_k^{\mathrm{exp}}\ge0,
        \qquad k=0,\ldots,N-1.
    \end{aligned}
    $$
    Here $P_k^{\mathrm{PV}}$ is a supplied input. Heating is
    re-optimised; both battery powers remain zero and battery states are absent.
    Exports accommodate surplus PV without a curtailment variable. The net site
    bill already includes electricity purchased for heating.
    ''')
    md('task-b-text',r'''
    ### Task B — formulate heating with PV (slide 10; 10 min including debrief)

    Create `grid_import` and `grid_export`, each with $N$ nonnegative entries.
    Complete the no-battery balance and the net-bill expression. Combine the
    balance with `heating_constraints` to form `problem_B`. The next cell supplies
    the solve, checks and comparison plots. Keep heating as a decision.
    ''')
    task('B',r'''
    grid_import=cp.Variable(N,nonneg=True,name='grid_import_kw')
    grid_export=cp.Variable(N,nonneg=True,name='grid_export_kw')
    balance_B=grid_import+pv_parameter==base_parameter+P+grid_export
    net_bill=dt*cp.sum(cp.multiply(import_price,grid_import)-cp.multiply(export_price,grid_export))
    problem_B=cp.Problem(cp.Minimize(net_bill),heating_constraints+[balance_B])
    ''',r'''
    # TODO B: create grid_import and grid_export (N nonnegative entries).
    # Define balance_B, net_bill and problem_B using the supplied Parameters.
    raise NotImplementedError('Task B: formulate the PV/no-battery site balance and bill.')
    ''')
    code('pv-solve-figure',r'''
    case_B=require_checked(solve_checked(problem_B,{'power_kw':P,'temperature_c':T,
        'grid_import_kw':grid_import,'grid_export_kw':grid_export,
        'charge_kw':np.zeros(N),'discharge_kw':np.zeros(N)},pv=pv_available,case=CASE_NAMES['B']))
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.5))
    for key,result in [('A',case_A),('B',case_B)]:
        axes[0].stairs(result['power_kw'],state_time,label=key+' heating',color=COLOURS[key])
        axes[1].stairs(result['grid_import_kw'],state_time,label=key+' import',color=COLOURS[key])
        axes[1].stairs(result['grid_export_kw'],state_time,label=key+' export',color=COLOURS[key],linestyle='--')
        axes[2].plot(state_time,result['temperature_c'],label=key,color=COLOURS[key])
    axes[0].stairs(pv_available,state_time,label='Available PV',color=COLOURS['pv'],linestyle=':')
    comfort_lines(axes[2])
    for ax,label in zip(axes,['Heating / PV (kW)','Grid exchange (kW)','Temperature (°C)']):
        clock_axis(ax);ax.set(xlabel='Time',ylabel=label);ax.legend(fontsize=9)
    save_figure(fig,'F72-11','A versus B')
    ''')
    interpretation('B','How does PV enter the balance and change the\nincentive for heating?',
        textwrap.dedent(r'''
    PV enters as a supplied generation profile. It reduces imports or increases
    exports through the balance. Using PV for heating forgoes the corresponding
    export revenue, which changes the marginal cost of heating. The computed
    optimum determines when heating shifts. For this day, heat-pump electricity
    falls from 20.045 to 19.513 kWh and heating shifts towards available PV.
    Almost all electricity is consumed locally.
    ''').strip())
    md('battery-text',r'''
    ## 4. Battery energy at interval boundaries (slides 12–16)

    Charge and discharge are powers in kW; $E_j$ is stored energy in kWh.
    There are $N=48$ interval powers and $N+1=49$ energy states. Add the following
    battery constraints to the heating and site equations:
    $$
    \begin{aligned}
    E_0&=E_{\mathrm{init}}=5\ \mathrm{kWh},\\
    E_{k+1}&=E_k+\Delta t\,\eta_{\mathrm{ch}}P_k^{\mathrm{ch}}
                    -\Delta t\,P_k^{\mathrm{dis}}/\eta_{\mathrm{dis}},
                    &&k=0,\ldots,N-1,\\
    0&\le P_k^{\mathrm{ch}}\le3\ \mathrm{kW},\qquad
    0\le P_k^{\mathrm{dis}}\le3\ \mathrm{kW},&&k=0,\ldots,N-1,\\
    0.5\ \mathrm{kWh}&\le E_j\le9.5\ \mathrm{kWh},&&j=0,\ldots,N,\\
    E_N&=E_0.
    \end{aligned}
    $$
    Both efficiencies are 0.95. The final equality preserves stored-energy
    service across the horizon. The battery recurrence multiplies power by
    $\Delta t$ to convert kW into kWh.

    The parameters describe an illustrative household-scale battery. The bounds
    provide a 9 kWh usable energy window; 3 kW limits the charging/discharging
    rate; 5 kWh starts at the window's midpoint. The efficiencies imply
    $0.95^2=90.25\%$ round-trip efficiency. For scale, the [Powerwall 2 UK
    datasheet](https://www.tesla.com/sites/default/files/pdfs/powerwall/Powerwall2ACDatasheet_EN_UK_feb2020.pdf)
    lists 13.5 kWh usable energy, 3.68/5 kW continuous power and 90% round-trip
    efficiency under its stated test conditions. These specifications provide a
    scale comparison for the teaching case. For a real installation, use its
    specified energy limits, inverter rating and efficiency data; set initial
    energy from the operating state.
    ''')
    code('battery-index',r'''
    fig,ax=plt.subplots(figsize=(12.5,3.5));ax.axis('off');ax.set(xlim=(-.25,4.3),ylim=(-.8,1.1))
    ax.plot(range(5),np.zeros(5),'o-',color='black',markersize=8)
    for j in range(5):ax.text(j,-.25,f'$E_{j}$ (kWh)',ha='center',fontsize=14)
    for k in range(4):
        ax.text(k+.5,.32,f'$P_{{{k}}}^{{ch}}, P_{{{k}}}^{{dis}}$ (kW)',ha='center',fontsize=12)
        ax.annotate('',xy=(k+1,.13),xytext=(k,.13),arrowprops={'arrowstyle':'<->'})
        ax.text(k+.5,.65,'$\\Delta t=0.5$ h',ha='center',fontsize=12)
    ax.text(2,-.65,'Illustration: 4 intervals / 5 states. Full case: 48 intervals / 49 states.',ha='center')
    save_figure(fig,'F72-03','Symbolic battery indexing','4 interval powers; 5 boundary energies')
    ''')
    md('task-c-text',r'''
    ### Task C — formulate storage (slide 15; 12 min including debrief)

    Create `charge`, `discharge` and `E`. Complete `battery_constraints`, including
    the recurrence, initial value, bounds at every boundary and terminal equality.
    Compute the energy added by 1 kW charging for half an hour and the energy
    removed by 1 kW discharging for half an hour. This is a formulation task;
    the battery is solved jointly with heating in Task D.
    ''')
    task('C',r'''
    charge = cp.Variable(N, nonneg=True, name='charge_kw')
    discharge = cp.Variable(N, nonneg=True, name='discharge_kw')
    E = cp.Variable(N + 1, name='stored_energy_kwh')
    battery_constraints = [
        E[0] == E_INIT,
        E[1:] == E[:-1] + dt * ETA_CHARGE * charge - dt * discharge / ETA_DISCHARGE,
        E >= E_MIN,
        E <= E_MAX,
        charge <= BATTERY_MAX,
        discharge <= BATTERY_MAX,
        E[-1] == E[0],
    ]
    one_kw_charge_kwh = dt * ETA_CHARGE
    one_kw_discharge_kwh = dt / ETA_DISCHARGE
    print(f'1 kW charge adds {one_kw_charge_kwh:.3f} kWh; discharge removes {one_kw_discharge_kwh:.4f} kWh.')
    ''',r'''
    # TODO C: create charge/discharge (N) and E (N+1), then battery_constraints.
    # E[:-1] holds E_k; E[1:] holds E_(k+1).
    # Compute one_kw_charge_kwh and one_kw_discharge_kwh for one half-hour.
    raise NotImplementedError('Task C: formulate battery dynamics, limits and terminal equality.')
    ''')
    interpretation('C','Why are the two half-hour energy changes\ndifferent?',
        textwrap.dedent(r'''
    Charging adds 0.475 kWh to storage. Delivering 0.5 kWh to the site removes
    approximately 0.5263 kWh from storage. Both directions incur losses, so
    restoring the initial stored energy requires more charging energy than the
    battery discharges to the site.
    ''').strip())
    md('joint-text',r'''
    ## 5. Construct the joint LP (slides 16–19)

    Minimise the single net site bill subject to heating constraints, battery
    constraints and the site balance. All expressions are affine, so the core is a
    linear program. The objective contains the electricity bill only. Storage can
    remain idle at 5 kWh when this achieves the minimum bill. We compare its
    operating value later under the same prices and comfort requirements.

    **After slide 16:** save the notebook and take the additional 15–20-minute
    break.

    ### Task D — solve jointly (slide 17; 12 min including debrief)

    Complete `balance_C`, assemble `problem_C` and call the supplied solve helper
    with the variables below. Read the resulting heating, battery and grid
    schedules alongside PV and prices. The helper preserves each schedule for the
    later A/B/C comparison.
    ''')
    task('D',r'''
    balance_C = grid_import + pv_parameter + discharge == base_parameter + P + charge + grid_export
    problem_C = cp.Problem(cp.Minimize(net_bill), heating_constraints + battery_constraints + [balance_C])
    site_variables = {
        'power_kw': P,
        'temperature_c': T,
        'grid_import_kw': grid_import,
        'grid_export_kw': grid_export,
        'charge_kw': charge,
        'discharge_kw': discharge,
        'energy_kwh': E,
    }
    candidate_C = solve_checked(problem_C, site_variables, pv=pv_available,
                                battery=True, case=CASE_NAMES['C'])
    case_C = require_checked(candidate_C)
    print(f"Case C net electricity bill: £{case_C['bill_gbp']:.3f}")
    ''',r'''
    # TODO D: define balance_C and problem_C (net bill only), then solve_checked.
    # Pass {'power_kw':P, 'temperature_c':T, 'grid_import_kw':grid_import,
    #       'grid_export_kw':grid_export, 'charge_kw':charge,
    #       'discharge_kw':discharge, 'energy_kwh':E}, pv=pv_available,
    #       battery=True, case=CASE_NAMES['C']; require_checked the result as case_C.
    raise NotImplementedError('Task D: assemble and solve the joint cost-only LP.')
    ''')
    code('joint-figures',r'''
    fig,axes=plt.subplots(3,1,sharex=True,figsize=(12.5,7))
    for values,label,colour in [(case_C['power_kw'],'Heating',COLOURS['C']),
                               (base_load,'Base',COLOURS['base']),(pv_available,'PV',COLOURS['pv'])]:
        axes[0].stairs(values,state_time,label=label,color=colour)
    for key,label,colour in [('charge_kw','Charge',COLOURS['charge']),('discharge_kw','Discharge',COLOURS['discharge'])]:
        axes[1].stairs(case_C[key],state_time,label=label,color=colour)
    axes[2].stairs(case_C['grid_import_kw'],state_time,label='Import',color=COLOURS['C'])
    axes[2].stairs(case_C['grid_export_kw'],state_time,label='Export',color='tab:green',linestyle='--')
    tariff_ax=axes[2].twinx();tariff_ax.stairs(import_prices,state_time,color='black',linestyle=':',label='Import tariff')
    tariff_ax.set_ylabel('£/kWh');tariff_ax.grid(False)
    for ax,label in zip(axes,['Site (kW)','Battery (kW)','Grid (kW)']):
        clock_axis(ax);ax.set_ylabel(label);ax.legend(loc='center left',bbox_to_anchor=(1.13,.5),fontsize=9)
    axes[2].set_xlabel('Interval time; powers plotted positive; dotted black line is import tariff')
    save_figure(fig,'F72-04',CASE_NAMES['C'])
    fig,ax=plt.subplots(figsize=(12.5,3.5))
    ax.plot(state_time,case_C['energy_kwh'],'o-',markersize=3,color=COLOURS['C'],label='C energy: half-hour boundaries')
    ax.axhline(E_MIN,color='black',linestyle='--',label='Energy limits');ax.axhline(E_MAX,color='black',linestyle='--')
    ax.scatter(state_time[[0,-1]],case_C['energy_kwh'][[0,-1]],color='black',zorder=3,label='Initial / final 5 kWh')
    clock_axis(ax);ax.set(xlabel='Boundary time',ylabel='Stored energy (kWh)');ax.legend(ncol=3,fontsize=10)
    save_figure(fig,'F72-05',CASE_NAMES['C'])
    ''')
    md('energy-trajectory',r'''
    The battery model is linear. Each marker is an optimised half-hour boundary
    state; straight segments join adjacent states without smoothing. With powers
    held constant within interval $k$, the slope is $\eta_{\mathrm{ch}}P_k^{\mathrm{ch}}
    -P_k^{\mathrm{dis}}/\eta_{\mathrm{dis}}$
    in kWh/h. This slope changes when charge or discharge changes, so several
    short segments can look curved at the full-day scale. The plotted shape
    follows the interval dispatch and resolution of the time grid. The recurrence
    uses fixed efficiencies and interval powers.
    ''')
    interpretation('D','What couples heating, PV, storage and grid\nexchange across time?',
        textwrap.dedent(r'''
    The site balance links heating, PV, storage and grid powers within each
    interval. Temperature and stored energy carry the effects of decisions into
    later intervals. The battery can charge from grid imports and PV. Aggregate
    flows describe the combined supply; identifying the origin of stored
    electricity requires an additional accounting rule. Interpret the charging
    trajectory together with the prices and available PV.
    ''').strip())
    md('operation-text',r'''
    ## 6. Electricity costs and battery energy (slides 20–24)

    Read the bill as the cost of imported electricity minus revenue from exports.
    The battery shifts electricity between intervals, but charging and discharging
    incur losses. Compare the energy entering and leaving storage alongside its
    initial and final energy. A lower bill can coincide with higher electricity
    purchases.

    ### Task E — explain the cost and energy flows (slide 19; 10 min including debrief)

    Calculate import cost, export revenue and the net bill for C. Calculate the
    electricity used to charge the battery and the electricity it returns to the
    site. With the battery finishing at its initial energy, their difference is
    the energy lost in charging and discharging.

    Then run the supplied end-of-day experiment below. It allows the battery to
    finish below 5 kWh. Compare the bill and remaining energy: how much of the
    apparent saving comes from leaving less energy for the following day?
    ''')
    task('E',r'''
    import_cost=float(dt*np.sum(import_prices*case_C['grid_import_kw']))
    export_revenue=float(dt*np.sum(export_prices*case_C['grid_export_kw']))
    net_bill_C=import_cost-export_revenue
    charging_energy=float(dt*np.sum(case_C['charge_kw']))
    discharged_energy=float(dt*np.sum(case_C['discharge_kw']))
    stored_energy_change=float(case_C['energy_kwh'][-1]-case_C['energy_kwh'][0])
    battery_losses=charging_energy-discharged_energy-stored_energy_change
    print(f'Import cost £{import_cost:.3f} − export revenue £{export_revenue:.3f} = net bill £{net_bill_C:.3f}.')
    print(f'Battery: {charging_energy:.3f} kWh in, {discharged_energy:.3f} kWh out, {battery_losses:.3f} kWh losses.')
    ''',r'''
    # TODO E: calculate import_cost, export_revenue and net_bill_C from case_C.
    # Calculate charging_energy, discharged_energy and stored_energy_change (kWh).
    # Use these to calculate battery_losses and explain the cost/energy flows.
    raise NotImplementedError('Task E: calculate the electricity bill and battery energy flows.')
    ''')
    code('operating-summary',r'''
    operation_summary=pd.DataFrame([
        ('Import cost',import_cost,'£'),('Export revenue',export_revenue,'£'),
        ('Net electricity bill',net_bill_C,'£'),('Battery charging energy',charging_energy,'kWh'),
        ('Battery discharged energy',discharged_energy,'kWh'),('Battery losses',battery_losses,'kWh'),
        ('Initial stored energy',case_C['energy_kwh'][0],'kWh'),
        ('Final stored energy',case_C['energy_kwh'][-1],'kWh')],columns=['Quantity','Value','Unit'])
    export_table(operation_summary,'C_operating_summary.csv')
    fig,ax=plt.subplots(figsize=(12.5,3.5))
    draw_table(ax,operation_summary,fontsize=12,scale=1.7,widths=[.55,.25,.20])
    ax.set_title('Case C: electricity costs and battery energy',fontsize=12)
    save_figure(fig,'F72-06',CASE_NAMES['C'],'Bill breakdown; battery energy flows and initial/final energy')
    ''')
    md('reuse-text',r'''
    The supplied function collects the equations already assembled above for
    controlled comparisons. It creates fresh heating decisions in every case.
    Disabling storage sets both battery powers to zero and omits its energy
    states. `terminal=False` removes only the final equality; `fixed_heating` and
    `peak_weight` are used only in the optional extensions.
    ''')
    code('reusable-site',r'''
    def build_site(pv,battery=False,terminal=True,fixed_heating=None,peak_weight=0.):
        if peak_weight<0:raise ValueError('Peak weight must be nonnegative.')
        heat=cp.Variable(N,name='heating_kw');temperature=cp.Variable(N+1,name='temperature_c')
        imp=cp.Variable(N,nonneg=True);exp=cp.Variable(N,nonneg=True)
        constraints=[temperature[0]==TRUE_T0,
            temperature[1:]==(1-b)*temperature[:-1]+b*outdoor+c*COP*heat,
            heat>=0,heat<=P_MAX,temperature[1:]>=lower,temperature[1:]<=upper]
        ch=np.zeros(N);dis=np.zeros(N)
        variables={'power_kw':heat,'temperature_c':temperature,'grid_import_kw':imp,'grid_export_kw':exp}
        if battery:
            ch=cp.Variable(N,nonneg=True);dis=cp.Variable(N,nonneg=True);energy=cp.Variable(N+1)
            constraints += [energy[0]==E_INIT,energy[1:]==energy[:-1]+dt*ETA_CHARGE*ch-dt*dis/ETA_DISCHARGE,
                energy>=E_MIN,energy<=E_MAX,ch<=BATTERY_MAX,dis<=BATTERY_MAX]
            if terminal:constraints += [energy[-1]==energy[0]]
            variables['energy_kwh']=energy
        variables.update(charge_kw=ch,discharge_kw=dis)
        constraints += [imp+pv+dis==base_load+heat+ch+exp]
        if fixed_heating is not None:constraints += [heat==np.asarray(fixed_heating).copy()]
        bill=dt*cp.sum(cp.multiply(import_prices,imp)-cp.multiply(export_prices,exp))
        objective=bill
        if peak_weight>0:
            peak=cp.Variable(nonneg=True)
            constraints += [imp<=peak]
            objective += peak_weight*peak
        return cp.Problem(cp.Minimize(objective),constraints),variables

    def optimise_site(pv,battery=False,terminal=True,fixed_heating=None,peak_weight=0.,case=''):
        problem,variables=build_site(pv,battery,terminal,fixed_heating,peak_weight)
        return solve_checked(problem,variables,pv=pv,battery=battery,terminal=terminal,
                             peak_weight=peak_weight,case=case)
    ''')
    code('end-of-day-experiment',r'''
    no_terminal=require_checked(optimise_site(pv_available,battery=True,terminal=False,case='Experiment: unrestricted final battery energy'))
    terminal_table=pd.DataFrame([{'Case':label,'Bill (£)':r['bill_gbp'],
        'Initial E (kWh)':r['energy_kwh'][0],'Final E (kWh)':r['energy_kwh'][-1]}
        for label,r in [('C: terminal equality',case_C),('No terminal equality',no_terminal)]])
    display(terminal_table.round(4));export_table(terminal_table,'terminal_comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5),gridspec_kw={'width_ratios':[1.1,1.3]})
    for label,result,colour in [('C: terminal equality',case_C,COLOURS['C']),('No terminal equality',no_terminal,'tab:red')]:
        axes[0].plot(state_time,result['energy_kwh'],label=label,color=colour)
    axes[0].axhline(E_MIN,color='0.5',linestyle='--');axes[0].axhline(E_MAX,color='0.5',linestyle='--')
    clock_axis(axes[0]);axes[0].set(xlabel='Boundary time',ylabel='Stored energy (kWh)');axes[0].legend(fontsize=9)
    draw_table(axes[1],terminal_table[['Case','Bill (£)','Final E (kWh)']],fontsize=10,widths=[.5,.23,.27])
    axes[1].set_title('Compare the bill and energy left for tomorrow',fontsize=11)
    save_figure(fig,'F72-07','C versus unrestricted final-energy experiment')
    ''')
    interpretation('E','What do the battery energy flows and the\nend-of-day experiment tell us about operating savings?',
        textwrap.dedent(r'''
    Charging and discharging losses mean the battery returns less electricity than
    it receives. It can still reduce the bill by shifting purchases to cheaper
    intervals. Allowing the final energy to fall below 5 kWh reduces the bill from
    £5.100 to £3.860 and leaves 0.5 kWh in storage. The extra £1.240 reduction
    comes with 4.5 kWh less energy for the next day. In the core A/B/C comparison,
    C restores its initial energy.
    ''').strip())
    md('comparison-text',r'''
    ## 7. Compare assets under common conditions (slides 22–25)

    | Case | Assets | Heating | Battery terminal condition |
    |---|---|---|---|
    | A: Heating only | Base demand and heat pump | Optimised | Not applicable |
    | B: Heating + PV | A plus original available PV | Re-optimised | Not applicable |
    | C: Heating + PV + battery | B plus storage | Jointly optimised | $E_N=E_0=5$ kWh |

    All cases share weather, model, base demand, tariffs, comfort bounds, initial
    temperature, horizon and thermal terminal treatment. A versus B identifies the
    operating effect of adding PV; B versus C identifies adding storage while
    heating adapts in both. Case A supplies the heating-only reference for these
    comparisons.

    ### Task F — assemble the evidence (slide 23; 14 min including debrief)

    Complete `performance_metrics(result)` from returned arrays and assemble
    `asset_comparison` for A/B/C. Report net bill, imported/exported energy,
    heat-pump electricity, actual maximum grid import, mean endpoint and final
    temperature, and lower/upper degree-hours. Battery fields are
    $E_0$, $E_N$ and throughput $\Delta t\sum_k(P_k^{ch}+P_k^{dis})$ in kWh.
    This throughput measures energy moved into and out of storage. A cycle count
    requires a separate definition. Mark absent battery fields as not applicable.
    Read cost, demand and comfort together when comparing the configurations.
    ''')
    task('F',r'''
    def performance_metrics(result):
        require_checked(result)
        endpoint_temperature=result['temperature_c'][1:]
        return {'case':result['case'],
            'bill_gbp':float(dt*np.sum(import_prices*result['grid_import_kw']-export_prices*result['grid_export_kw'])),
            'import_kwh':float(dt*np.sum(result['grid_import_kw'])),
            'export_kwh':float(dt*np.sum(result['grid_export_kw'])),
            'heat_pump_kwh':float(dt*np.sum(result['power_kw'])),
            'peak_import_kw':float(np.max(result['grid_import_kw'])),
            'mean_endpoint_temperature_c':float(endpoint_temperature.mean()),'final_temperature_c':float(endpoint_temperature[-1]),
            'below_band_degree_hours':float(dt*np.maximum(lower-endpoint_temperature,0.).sum()),
            'above_band_degree_hours':float(dt*np.maximum(endpoint_temperature-upper,0.).sum()),
            'initial_energy_kwh':float(result['energy_kwh'][0]) if result['battery'] else np.nan,
            'final_energy_kwh':float(result['energy_kwh'][-1]) if result['battery'] else np.nan,
            'battery_throughput_kwh':float(dt*np.sum(result['charge_kw']+result['discharge_kw'])) if result['battery'] else np.nan}
    core_cases={'A':case_A,'B':case_B,'C':case_C}
    asset_comparison=pd.DataFrame([performance_metrics(result) for result in core_cases.values()])
    display(asset_comparison.round(4))
    ''',r'''
    # TODO F: define performance_metrics(result) using the quantities above.
    # Required columns: case, bill_gbp, import_kwh, export_kwh, heat_pump_kwh,
    # peak_import_kw, mean_endpoint_temperature_c, final_temperature_c,
    # below_band_degree_hours, above_band_degree_hours, initial_energy_kwh,
    # final_energy_kwh, battery_throughput_kwh. Use np.nan for absent storage.
    # Define core_cases={'A':case_A,'B':case_B,'C':case_C} and asset_comparison.
    raise NotImplementedError('Task F: compute and interpret matched A/B/C performance metrics.')
    ''')
    code('comparison-figures',r'''
    export_table(asset_comparison,'asset_comparison_metrics.csv')
    fig=plt.figure(figsize=(12.5,3.5));gs=fig.add_gridspec(2,4,width_ratios=[2.4,1,1,1])
    imp_ax=fig.add_subplot(gs[0,0]);context_ax=fig.add_subplot(gs[1,0],sharex=imp_ax)
    for key,result in core_cases.items():
        imp_ax.stairs(result['grid_import_kw'],state_time,label=key,color=COLOURS[key])
    imp_ax.set_ylabel('Import (kW)');imp_ax.legend(ncol=3,fontsize=9);clock_axis(imp_ax)
    context_ax.stairs(pv_available,state_time,color=COLOURS['pv']);context_ax.set(ylabel='PV (kW)',xlabel='Interval time')
    price_ax=context_ax.twinx();price_ax.stairs(import_prices,state_time,color='black',linestyle=':')
    price_ax.set_ylabel('£/kWh');price_ax.grid(False);clock_axis(context_ax)
    for column,(metric,label) in enumerate([('bill_gbp','Bill (£)'),('import_kwh','Import (kWh)'),('peak_import_kw','Peak (kW)')],1):
        ax=fig.add_subplot(gs[:,column]);ax.bar(list(core_cases),asset_comparison[metric],color=[COLOURS[k] for k in core_cases])
        ax.set_ylabel(label);ax.set_xlabel('Case')
    save_figure(fig,'F72-12','A / B / C')
    compact=asset_comparison[['case','bill_gbp','import_kwh','export_kwh','peak_import_kw',
                              'below_band_degree_hours','above_band_degree_hours']].copy()
    compact['case']=list(core_cases)
    compact.columns=['Case','Bill (£)','Import (kWh)','Export (kWh)','Peak (kW)','Below (°C h)','Above (°C h)']
    export_table(compact,'F72-10_native_table.csv')
    fig,ax=plt.subplots(figsize=(12.5,3.5));draw_table(ax,compact,fontsize=12,scale=2.4)
    ax.set_title('A: heating only   |   B: heating + PV   |   C: heating + PV + battery',fontsize=12)
    save_figure(fig,'F72-10','A / B / C','3 cases; 6 common performance metrics')
    pv_value=case_A['bill_gbp']-case_B['bill_gbp']
    storage_value=case_B['bill_gbp']-case_C['bill_gbp']
    print(f'Observed operating bill differences: A − B = £{pv_value:.4f}; B − C = £{storage_value:.4f}.')
    ''')
    interpretation('F','Explain the PV and storage comparisons using\nbill, energy, peak and comfort. What can this day establish?',
        textwrap.dedent(r'''
    A versus B adds PV, and B versus C adds storage while heating is re-optimised.
    Storage can remain idle when this achieves the minimum bill. Compare
    electricity, peak import and comfort separately from the bill: losses, timing
    and temperature trajectories can change their rankings.

    For this day, A/B/C bills are £7.815, £5.703 and £5.100. Adding PV reduces the
    bill by £2.112; adding storage reduces it by a further £0.603. Storage
    increases imports from 23.460 to 23.635 kWh while reducing actual peak import
    from 5.660 to 4.919 kW. All three cases have zero reported comfort
    degree-hours and finish at 20°C. C returns to 5 kWh with 10.410 kWh
    throughput. These results describe one-day operation with known inputs.
    Investment returns and robustness require further evaluation.
    ''').strip())
    md('mpc-boundary',r'''
    Temperatures are evaluated at half-hour boundaries. The length of the planning
    horizon and the energy required at its end affect the operating schedule.
    PV is a supplied input; controllable heating and battery powers provide
    flexibility within their constraints. A thermostat reacts to temperature;
    MPC repeatedly updates a model-based plan as measurements and forecasts arrive.
    Week 8 retains its thermal-only core; joint site MPC is an extension.
    ''')
    md('optional-text',r'''
    ## Optional extensions (slides 28–30; outside core timing)

    **Fixed heating versus joint scheduling.** Fix heating to B's checked PV-only
    optimum and optimise storage. Compare with C under identical physical
    constraints and the same net-bill objective. Joint scheduling has the option
    to retain B's heating schedule or adapt it if doing so reduces the bill.

    **Bill and peak.** For case C add $\lambda P^{peak}$ with $P_k^{imp}\le P^{peak}$ and
    $\lambda=0.35$ £/kW. This illustrative weight assigns a cost to peak import
    in the objective. Report bill, actual peak, weighted term and objective
    separately. At zero weight, calculate the actual peak directly from imports.
    ''')
    code('optional-solves',r'''
    fixed_heating=require_checked(optimise_site(pv_available,battery=True,
        fixed_heating=case_B['power_kw'],case='Optional: fixed B heating + battery'))
    peak_weighted=require_checked(optimise_site(pv_available,battery=True,peak_weight=.35,
        case='Optional: C with peak weight'))
    fixed_comparison=pd.DataFrame([performance_metrics(r) for r in [fixed_heating,case_C]])
    display(fixed_comparison.round(4));export_table(fixed_comparison,'fixed_vs_joint_metrics.csv')
    objective_table=pd.DataFrame([{'Case':name,'Bill (£)':r['bill_gbp'],'Actual peak (kW)':r['actual_peak_kw'],
        'Peak term (£)':r['peak_term_gbp'],'Objective (£)':r['recomputed_objective']}
        for name,r in [('C: cost only',case_C),('C: peak weight',peak_weighted)]])
    display(objective_table.round(4));export_table(objective_table,'cost_peak_metrics.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5),gridspec_kw={'width_ratios':[1,1.6]})
    for label,r,colour in [('Cost only',case_C,COLOURS['C']),('Peak weight',peak_weighted,COLOURS['peak'])]:
        axes[0].stairs(r['grid_import_kw'],state_time,label=label,color=colour)
    clock_axis(axes[0]);axes[0].set(xlabel='Interval time',ylabel='Grid import (kW)');axes[0].legend(fontsize=10)
    draw_table(axes[1],objective_table,fontsize=9,widths=[.24,.16,.20,.20,.20])
    axes[1].set_title('Same assets; λ = 0 or 0.35 £/kW',fontsize=12)
    save_figure(fig,'F72-08','C cost-only versus peak-weighted')
    ''')
    md('event-text',r'''
    **Event evaluation.** Compare the peak-weighted schedule with the same-asset
    cost-only C baseline for $16{:}00\le t<19{:}00$: exactly six half-hour intervals.
    The signed difference is baseline-minus-candidate imported kWh. Retain
    pre-event and post-event context. The objective targets the daily peak; the
    event-energy difference can be positive, zero or negative.
    ''')
    code('optional-event',r'''
    event_start=interval_start[0].normalize()+pd.Timedelta(hours=16)
    event_end=interval_start[0].normalize()+pd.Timedelta(hours=19)
    masks={'Pre':interval_start<event_start,'Event':(interval_start>=event_start)&(interval_start<event_end),
           'Post':interval_start>=event_end}
    assert int(masks['Event'].sum())==6
    event_table=pd.DataFrame([{'Period':name,'Intervals':int(mask.sum()),
        'Cost-only (kWh)':float(dt*case_C['grid_import_kw'][mask].sum()),
        'Peak-weighted (kWh)':float(dt*peak_weighted['grid_import_kw'][mask].sum()),
        'Difference (kWh)':float(dt*(case_C['grid_import_kw'][mask]-peak_weighted['grid_import_kw'][mask]).sum())}
        for name,mask in masks.items()])
    display(event_table.round(4));export_table(event_table,'event_flexibility.csv')
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5),gridspec_kw={'width_ratios':[1,1.5]})
    for label,r,colour in [('Cost only',case_C,COLOURS['C']),('Peak weight',peak_weighted,COLOURS['peak'])]:
        axes[0].stairs(r['grid_import_kw'],state_time,label=label,color=colour)
    axes[0].axvspan(event_start,event_end,color='0.8',alpha=.5,label='16:00–19:00')
    clock_axis(axes[0]);axes[0].set(xlabel='Interval time',ylabel='Grid import (kW)');axes[0].legend(fontsize=9)
    draw_table(axes[1],event_table.drop(columns='Intervals'),fontsize=10,widths=[.16,.26,.29,.29])
    axes[1].set_title('Signed cost-only minus peak-weighted energy',fontsize=11)
    save_figure(fig,'F72-09','Same-asset optional event comparison','6 event intervals; full-day context')
    ''')
    md('optional-interpretation',r'''
    Read the separately reported bill and peak before ranking the optional schedules.
    For this case the peak weight reduces actual maximum import, with a small
    increase in bill. The event table shows essentially zero imported energy in
    both schedules during 16:00–19:00, so there is no meaningful event reduction.
    Values in the figures are rounded
    for readability; the CSV files retain full precision and unaltered powers.
    ''')
    md('extensions',r'''
    EVs, degradation, curtailment and explicit binary flow modes change the model
    and remain discussion extensions. An overnight EV requires a horizon that
    includes both arrival and departure; a midnight-to-midnight mask alone does
    not describe an 18:00-to-07:00 connection correctly. Battery throughput
    summarises energy movement; estimating degradation requires an additional
    model.
    ''')
    code('exports',r'''
    all_results={**core_cases,'no_terminal':no_terminal,'fixed_heating':fixed_heating,'peak_weighted':peak_weighted}
    for key,result in all_results.items():
        export_schedule(result,key)
    if EXPORT_ARTEFACTS:
        pd.DataFrame(figure_manifest).sort_values('id').to_csv(ROOT/'teaching/week7_day2_figure_manifest.csv',index=False)
        provenance={'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
            'period_start':str(interval_start[0]),'period_end':str(state_time[-1]),'N':N,'dt_hours':dt,
            'initial_temperature_c':TRUE_T0,'battery':{'eta_charge':ETA_CHARGE,'eta_discharge':ETA_DISCHARGE,
                'initial_kwh':E_INIT,'lower_kwh':E_MIN,'upper_kwh':E_MAX,'power_limit_kw':BATTERY_MAX},
            'solver':SOLVER,'throughput_definition':'dt * sum(charge + discharge), kWh; not cycles',
            'cases':{key:{field:result[field] for field in ['case','battery','terminal','peak_weight']} for key,result in all_results.items()},
            'data_provenance':'Supplied synthetic reference-building case; no profile changes; known future planning inputs.'}
        (OUTPUT_DIR/'run_metadata.json').write_text(json.dumps(provenance,indent=2),encoding='utf8')
    ''')
    md('references',r'''
    ## References and source credits (slide 27)

    - [Reference-building data README](data/reference_building/README.md): synthetic teaching case and model limitations.
    - Boyd, S. and Vandenberghe, L. (2004), *Convex Optimization*, Cambridge University Press.
    - Diamond, S. and Boyd, S. (2016), “CVXPY: A Python-Embedded Modeling Language for Convex Optimization”, *Journal of Machine Learning Research*, 17(83), 1–5.
    ''')
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                                  'language_info':{'name':'python','version':'3'}})


def write_metadata():
    teaching=ROOT/'teaching';teaching.mkdir(exist_ok=True)
    with (teaching/'week7_day2_task_manifest.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['task_id','slide','minutes_including_debrief','cell_id','notebook'])
        writer.writeheader()
        for letter,slide,minutes in TASKS:
            writer.writerow(dict(task_id=letter,slide=slide,minutes_including_debrief=minutes,
                                 cell_id='w72-task-'+letter.lower(),notebook=STEM+'.ipynb'))
    with (teaching/'week7_day2_schedule.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['slide','block','core_minutes','role','task','break_after_minutes'])
        writer.writeheader()
        for slide in range(1,37):
            task=next((letter for letter,s,_ in TASKS if s==slide),'')
            writer.writerow(dict(slide=slide,block=(slide-1)//8+1 if slide<=32 else 0,
                core_minutes=SLIDE_MINUTES[slide-1] if slide<=32 else 0,
                role='Task and debrief' if task else ('Core' if slide<=32 else 'References' if slide==33 else 'Optional'),
                task=task,break_after_minutes='15–20 additional' if slide==16 else ''))


if __name__=='__main__':
    for solution in [False,True]:
        nbf.write(make_notebook(solution),ROOT/(STEM+('_solution' if solution else '')+'.ipynb'))
    write_metadata()
    print('Built independent Tutorial 7.2 pair: six tasks; 70 task minutes within 200 teaching minutes.')
