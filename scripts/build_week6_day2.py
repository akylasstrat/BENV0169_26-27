"""Scoped Tutorial 6.2 authoring. Reconcile live manual edits before rebuilding."""
from pathlib import Path
import csv
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STEM = 'tutorial_6_2_rc_model_identification'
TASKS = [('A','I1',7,12),('B','I2',13,12),('C','I3',15,8),
         ('D','I4',19,12),('E','I5',21,8),
         ('R1','I7',25,10),('R2','I7',27,15),('R3','I7',29,15)]
SLIDE_MINUTES = [3,3,6,5,6,5,12,5,5,6,5,5,12,6,8,8,5,5,12,6,8,4,5,5,10,3,15,2,15,3,2]
TEACHING_BLOCKS = [(1,9),(10,16),(17,24),(25,31)]
FIGURES = [
 ('F62-01','w6d2_F01_identification_data','4','I0','1344 observations; 28 days','Synthetic temperatures, useful heat and solar input; first 21 days fit, last seven validate.'),
 ('F62-02','w6d2_F02_transition_rows','5','I1','6 example transitions','Actual source observations aligned to the next half-hour target.'),
 ('F62-03','w6d2_F03_fit_validation_timing','6;8','I1','','Within-period transitions exclude the fit-to-validation crossing.'),
 ('F62-04','w6d2_F04_fit_comparison','14','I2','1007 fitting transitions','OLS and bounded fits use identical rows and predictors; active bounds are reported.'),
 ('F62-05','w6d2_F05_prediction_protocols','18','I4','','Measured-state resets versus one initial measurement and recursive predictions.'),
 ('F62-06','w6d2_F06_validation_trajectories','20','I4','335 scored targets; 336 plotted observations','Seven-day conditional validation with the initial state excluded from scores.'),
 ('F62-07','w6d2_F07_validation_scores','22','I5','335 common targets per method','Actual-minus-predicted scores; one-step and seven-day rollout use different information sets.'),
 ('F62-08','w6d2_F08_residual_diagnostics','24','I6','335 one-step residuals','Observed-minus-predicted residuals at the temperature observation time, paired with heating input during the preceding interval; associations are diagnostic, not causal proof.'),
 ('F62-10','w6d2_F10_measured_house_transfer','30','I7','Common June hold-out targets','Ten-minute one-step predictions with November and June coefficients on identical held-out June observations.'),
 ('F62-11','w6d2_F11_fit_period_sensitivity','28','I7','Two fitting periods','Same three predictors and coefficient bounds fitted to November and June training measurements; estimates describe training-period sensitivity.'),
 ('F62-12','w6d2_F12_house_measurements','26','I7','720 observations per period','Original indoor/outdoor temperatures, total power and solar irradiance; missing readings retained.'),
]


def make_notebook(solution=False):
    cells=[]
    def md(key,source):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(),id='w62-'+key))
    def code(key,source,tags=None):cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(),id='w62-'+key,metadata={'tags':tags or ['supplied']}))
    def task(letter,answer,scaffold):
        code('task-'+letter.lower(),answer if solution else scaffold,['task',letter])

    md('purpose',r'''
    # Tutorial 6.2: RC model identification and validation

    We first fit an RC model to synthetic measurements and compare one-step
    predictions with a free rollout. We then fit two models to occupied-house
    measurements, compare their coefficients and test seasonal transfer.

    **Outcomes:** align transitions; fit bounded coefficients; recover effective
    $R,C$ and time constant values; include effective solar gains; compare
    seasonal coefficients and predictions on a shared summer hold-out.

    | Block | Slides | Minutes | Focus |
    |---|---|---:|---|
    | 1 | 1–9 | 50 | Measurements, timing and Task A |
    | 2 | 10–16 | 50 | Estimation and Tasks B–C |
    | Break | after 16 | 15–20 additional | Save and pause |
    | 3 | 17–24 | 50 | Validation and Tasks D–E |
    | 4 | 25–31 | 50 | Real-house experiment: Tasks R1–R3 |

    Tasks A–E take 52 minutes; Tasks R1–R3 take 40 minutes within the
    200-minute session. All plots are supplied. Run from the repository root
    using the local datasets and the course Python environment.
    ''')
    md('i0',r'''
    ## I0. Data, units and purpose (slides 3–4)

    In Week 5, calibration adjusted uncertain EnergyPlus inputs to improve
    agreement with operational data. Here, system identification estimates RC
    model coefficients from temperature and heating data.

    The frozen [reference data](data/reference_building/README.md) contain 28 days
    of noisy synthetic indoor measurements at 30-minute intervals. The row at
    $t_k$ contains measured $T_k$, outdoor $T_{o,k}$ and useful heat $q_{th,k}$;
    latter inputs act during the interval from $t_k$ to $t_{k+1}$.

    The one-state energy balance and its forward-Euler discretisation are

    $$C\frac{dT}{dt}=\frac{T_o-T}{R}+q_{th},$$

    $$\Delta T_k:=T_{k+1}-T_k=b(T_{o,k}-T_k)+cq_{th,k}+\epsilon_k,
    \qquad b=\frac{\Delta t}{RC},\quad c=\frac{\Delta t}{C}.$$

    Here $R$ is in K/kW, $C$ in kWh/K, $\Delta t=0.5$ h, temperatures in °C
    and heat input in kW. The residual $\epsilon_k$ includes measurement noise
    and dynamics omitted by the fitted model. We fit the two coefficients with
    the intercept fixed at zero. Section I7 adds an explicit solar-gain term.

    The supplied synthetic generator has two thermal states, solar/internal
    gains and temperature-dependent heating. Its air sensor noise has standard
    deviation 0.06°C; process noise has standard deviation 0.012°C per air-state
    interval and one third of this value for the fabric state. We use indoor
    measurements and recorded inputs to fit the simpler one-state model.
    Measurement noise in $T_k$ enters both the predictor and the target, so
    least-squares estimates can differ from physical fabric parameters.

    We keep the synthetic core for a controlled timing example. The real-house
    measurements are already supplied locally in `data/tutorial_6_2`; see the
    [local data README](data/tutorial_6_2/README.md) and
    [Hollick and Wingfield (2018), dataset DOI](https://doi.org/10.14324/000.ds.10087216).
    Section I7 develops the seasonal experiment as part of the main tutorial.
    ''')
    code('setup','''
    from pathlib import Path
    import hashlib
    import json
    import textwrap
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch
    from scipy.optimize import lsq_linear
    from IPython.display import display

    ROOT=Path.cwd()
    DATA_PATH=ROOT/"data/reference_building/identification_data.csv"
    if not DATA_PATH.is_file() or not (ROOT/"data/tutorial_6_2/10mins_solpap_2016.csv").is_file():
        raise FileNotFoundError("Run from the repository root containing data/reference_building and data/tutorial_6_2.")
    FIGURE_DIR=ROOT/"figures/week6_day2"
    OUTPUT_DIR=ROOT/"teaching/week6_day2_outputs"
    EXPORT_ARTEFACTS=True
    # None exports every figure; set IDs when refreshing selected lecture figures.
    FIGURE_EXPORT_IDS=None
    EXTRA_EXPORT_NAMES=None
    dt=.5
    plt.rcParams.update({"figure.dpi":110,"font.size":12,"axes.grid":True,
        "grid.alpha":.2,"svg.fonttype":"none","savefig.facecolor":"white"})
    COLOURS={"measured":"black","one_step":"tab:blue","rollout":"tab:orange",
             "outdoor":"0.5","fit":"tab:blue","validation":"tab:green"}
    ''')
    md('helpers-text',r'''
    ### Supplied plotting/export helpers

    Set `EXPORT_ARTEFACTS=False` to avoid writing files. No calculation reads an
    exported answer. The table helper rounds only its display copy.
    ''')
    code('helpers','FIGURE_SPECS='+repr(FIGURES)+'\n'+textwrap.dedent('''
    figure_manifest=[]
    def export_table(frame,name,index=False):
        if EXPORT_ARTEFACTS:
            OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
            frame.to_csv(OUTPUT_DIR/name,index=index)

    def save_figure(fig,figure_id,source_files=(),period="",units="",case="",action="Supplied",counts=None):
        spec=next(s for s in FIGURE_SPECS if s[0]==figure_id)
        diagram=figure_id in {"F62-03","F62-05"}
        if not source_files and not diagram:source_files=("scripts/build_week6_day2.py",)
        fig.tight_layout()
        if EXPORT_ARTEFACTS:
            FIGURE_DIR.mkdir(parents=True,exist_ok=True)
            if FIGURE_EXPORT_IDS is None or figure_id in FIGURE_EXPORT_IDS:
                for ext in ["png","svg"]:fig.savefig(FIGURE_DIR/f"{spec[1]}.{ext}",dpi=220,bbox_inches="tight")
            figure_manifest.append({"id":figure_id,"png":f"figures/week6_day2/{spec[1]}.png",
                "svg":f"figures/week6_day2/{spec[1]}.svg","slide":spec[2],"section":spec[3],
                "placement":"Full-width","source":"instructional diagram" if diagram else ";".join(source_files),
                "source_files":";".join(source_files),"source_sha256":";".join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files),
                "period":period,"units":units,"sample_transition_count":counts or spec[4],
                "model_case":case,"student_action":action,"caption":spec[5]})
        plt.show();plt.close(fig)

    def time_axis(ax):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=8))

    def legend_above(ax,ncol=3):
        ax.legend(loc="lower left",bbox_to_anchor=(0,1.01),ncol=ncol,fontsize=10)
        if ax.get_title():ax.set_title(ax.get_title(),pad=34)

    def table_figure(frame,widths=None,fontsize=12,height=3.5):
        shown=frame.copy()
        for column in shown.select_dtypes(include="floating"):
            shown[column]=shown[column].map(lambda x:f"{x:.6g}")
        shown=shown.map(lambda x:textwrap.fill(str(x),24))
        fig,ax=plt.subplots(figsize=(12,height));ax.axis("off")
        tab=ax.table(cellText=shown.values,colLabels=shown.columns,cellLoc="center",colWidths=widths,loc="center")
        tab.auto_set_font_size(False);tab.set_fontsize(fontsize)
        for (row,col),cell in tab.get_celld().items():
            cell.set_edgecolor("0.8");cell.set_height(.19 if row==0 else .17)
            if row==0:cell.set_facecolor("0.92")
        return fig
    ''').strip())
    code('read-data',r'''
    data=pd.read_csv(DATA_PATH,index_col="timestamp",parse_dates=True)
    required=["indoor_temperature_c","outdoor_temperature_c","delivered_heat_kw"]
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Source timestamps must be unique and sorted.")
    if not data.index.equals(pd.date_range(data.index[0],periods=len(data),freq="30min")):
        raise ValueError("Source must have a complete half-hour grid; do not bridge missing intervals.")
    if not np.isfinite(data[required].to_numpy()).all():
        raise ValueError("Inspect invalid required source readings before fitting.")
    split_time=pd.Timestamp("2025-01-27")
    fit_observations=data.loc[data.index<split_time].copy()
    validation_observations=data.loc[data.index>=split_time].copy()
    source_audit=pd.DataFrame({"period":["fit","validation"],"observations":[len(fit_observations),len(validation_observations)],
        "start":[fit_observations.index[0],validation_observations.index[0]],
        "end":[fit_observations.index[-1],validation_observations.index[-1]]})
    display(source_audit);display(data.head())
    export_table(source_audit,"source_audit.csv")
    fig,axes=plt.subplots(3,1,figsize=(12, 7),sharex=True)
    axes[0].plot(data.index,data.indoor_temperature_c,color="black",label="Indoor (synthetic)")
    axes[0].plot(data.index,data.outdoor_temperature_c,color="0.5",label="Outdoor")
    for ax in axes:
        ax.axvspan(data.index[0],split_time,color=COLOURS["fit"],alpha=.08,label="Fit: 21 days" if ax is axes[0] else None)
        ax.axvspan(split_time,data.index[-1],color=COLOURS["validation"],alpha=.08,label="Validate: 7 days" if ax is axes[0] else None)
        ax.axvline(split_time,color="0.4",linestyle="--");ax.set_xlim(data.index[0],data.index[-1])
    axes[0].set(ylabel="Temp. (°C)",title=f"{len(data)} synthetic samples; 30 min")
    legend_above(axes[0],2);axes[0].set_title(axes[0].get_title(),pad=50)
    axes[1].step(data.index,data.delivered_heat_kw,where="post",color="tab:orange")
    axes[1].set(ylabel="Useful heat (kW)",title="Heating input")
    axes[2].plot(data.index,data.solar_irradiance_w_m2,color="tab:green")
    axes[2].set(ylabel="Solar (W/m²)",title="Irradiance")
    for ax in axes:
        ax.set_xlabel("Observation time")
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    save_figure(fig,"F62-01",["data/reference_building/identification_data.csv"],str(data.index[0])+" to "+str(data.index[-1]),"°C;kW;W/m²;h","Frozen two-state synthetic observations")
    ''')
    md('i1',r'''
    ## I1. Task A — align observations and transitions (slide 7; 12 min)

    The fit period has 1008 observations (6–26 January); validation has 336
    (27 January–2 February). Build transitions **within each partition**.
    Exclude 26 January 23:30 → 27 January 00:00: its target initialises validation.

    Complete the target timestamp, next measured temperature, temperature change
    and outdoor-minus-indoor difference. Retain only finite consecutive pairs
    exactly 0.5 h apart. Build fitting `X_fit`, `y_fit` in the stated column order.
    Here parameter calibration means fitting, unlike Week 4 residual calibration.
    ''')
    task('A','''
    transition_tables={}
    for label,observations in [("fit",fit_observations),("validation",validation_observations)]:
        table=observations[required].copy()
        table["predictor_time"]=observations.index
        table["target_time"]=observations.index.to_series().shift(-1)
        table["next_temperature_c"]=table.indoor_temperature_c.shift(-1)
        table["temperature_change_c"]=table.next_temperature_c-table.indoor_temperature_c
        table["temperature_difference_c"]=table.outdoor_temperature_c-table.indoor_temperature_c
        finite=np.isfinite(table[required+["next_temperature_c"]]).all(axis=1)
        consecutive=(table.target_time-table.predictor_time).eq(pd.Timedelta(hours=dt))
        transition_tables[label]=table.loc[finite & consecutive].copy()
    fit_transitions=transition_tables["fit"]
    validation_transitions=transition_tables["validation"]
    FEATURE_COLUMNS=["temperature_difference_c","delivered_heat_kw"]
    X_fit=fit_transitions[FEATURE_COLUMNS].to_numpy()
    y_fit=fit_transitions.temperature_change_c.to_numpy()
    ''','''
    transition_tables={}
    for label, observations in [("fit", fit_observations), ("validation", validation_observations)]:
        table = observations[required].copy()
        table["predictor_time"] = observations.index
        table["target_time"] = observations.index.to_series().shift(-1)
        # TODO A: obtain next_temperature_c using shift(-1), then form the two differences.
        table["next_temperature_c"] = np.nan
        table["temperature_change_c"] = np.nan
        table["temperature_difference_c"] = np.nan
        finite = np.isfinite(table[required + ["next_temperature_c"]]).all(axis=1)
        consecutive = (table.target_time - table.predictor_time).eq(pd.Timedelta(hours=dt))
        transition_tables[label] = table.loc[finite & consecutive].copy()
    fit_transitions = transition_tables["fit"]
    validation_transitions = transition_tables["validation"]
    FEATURE_COLUMNS = ["temperature_difference_c", "delivered_heat_kw"]
    # TODO A: extract X_fit and y_fit from fit_transitions as NumPy arrays.
    raise NotImplementedError("Task A: align the within-period half-hour transitions.")
    ''')
    code('transition-results','''
    transition_counts=pd.DataFrame({"period":["fit","validation"],"N_transitions":[len(fit_transitions),len(validation_transitions)]})
    display(transition_counts);export_table(transition_counts,"transition_counts.csv")
    example_rows=fit_transitions.head(6)[["predictor_time","indoor_temperature_c","outdoor_temperature_c","delivered_heat_kw","target_time","next_temperature_c"]].copy()
    export_table(example_rows,"example_transitions.csv")
    shown=example_rows.copy()
    for column in ["predictor_time","target_time"]:shown[column]=shown[column].dt.strftime("%d %b %H:%M")
    shown.columns=["State time","T_k (°C)","To_k (°C)","q_k (kW)","Target time","T_k+1 (°C)"]
    fig=table_figure(shown,[.22,.13,.13,.13,.22,.17],11,4)
    save_figure(fig,"F62-02",["data/reference_building/identification_data.csv"],"First six within-fit transitions","°C;kW;timestamp","Actual source rows","Task A alignment")
    ''')
    code('timing-diagram',r'''
    fig,ax=plt.subplots(figsize=(12,4));ax.set(xlim=(0,10),ylim=(0,5));ax.axis("off")
    for x,label in [(1,"Tₖ at tₖ"),(4,"Tₖ₊₁ at tₖ₊₁"),(7,"Next boundary")]:
        ax.plot(x,3.4,"o",color="tab:blue",ms=10);ax.text(x,3.9,label,ha="center",fontsize=14)
    ax.annotate("",xy=(4,3.4),xytext=(1,3.4),arrowprops={"arrowstyle":"->"})
    ax.text(2.5,2.8,"Toₖ, qₖ act during this interval",ha="center",fontsize=13)
    ax.text(2.5,2.1,"Regression target: Tₖ₊₁ − Tₖ",ha="center",fontsize=14)
    ax.axvline(6,color="tab:red",linestyle="--",ymin=.08,ymax=.42)
    ax.text(3.5,.65,"26 Jan 23:30",ha="center",fontsize=13)
    ax.text(7.8,.65,"27 Jan 00:00\nValidation initial state",ha="center",fontsize=13)
    ax.annotate("Excluded crossing",xy=(7,1.4),xytext=(3.5,1.4),color="tab:red",fontsize=13,
                arrowprops={"arrowstyle":"->","color":"tab:red"})
    ax.text(6,1.8,"Split",ha="center",fontsize=12,color="tab:red")
    save_figure(fig,"F62-03",units="timestamp;°C;kW",case="Within-period transition contract")
    ''')
    md('i2',r'''
    ## I2. OLS and Task B — bounded coefficients (slide 13; 12 min)

    The supplied unconstrained fit minimises $\|X\theta-y\|^2$ on fitting rows
    only, where $\theta=[b,c]^T$. Use the **same X,y** for the bounded fit.
    Bounds $b\in[10^{-5},0.4]$ and $c\in[10^{-5},0.5]$ specify the allowed
    coefficient range at $\Delta t=0.5$ h; $c$ has K/kW units.
    $0<b\leq1$ gives non-oscillatory Euler relaxation; stability permits $0<b<2$.

    Complete `lsq_linear`, check success and obtain $b,c$. Compare the two
    fits using the supplied bounds and inputs. Equal coefficients are a valid
    result when the unconstrained estimate lies inside the bounds.
    ''')
    code('ols','''
    ols_coefficients=np.linalg.lstsq(X_fit,y_fit,rcond=None)[0]
    LOWER_BOUNDS=np.array([1e-5,1e-5])
    UPPER_BOUNDS=np.array([.4,.5])
    ''')
    task('B','''
    bounded_fit=lsq_linear(X_fit,y_fit,bounds=(LOWER_BOUNDS,UPPER_BOUNDS))
    if not bounded_fit.success:raise RuntimeError(bounded_fit.message)
    b,c=bounded_fit.x
    ''','''
    # TODO B: call lsq_linear on X_fit,y_fit with the supplied bounds; check success; obtain b,c.
    raise NotImplementedError("Task B: fit bounded coefficients on fitting data only.")
    ''')
    code('fit-comparison','''
    fit_comparison=pd.DataFrame([
        {"method":"OLS","b":ols_coefficients[0],"c_K_per_kW":ols_coefficients[1],"N_fit":len(y_fit),
         "deltaT_RMSE_K":np.sqrt(np.mean((y_fit-X_fit@ols_coefficients)**2)),"active_bounds":"not constrained"},
        {"method":"Bounded","b":b,"c_K_per_kW":c,"N_fit":len(y_fit),
         "deltaT_RMSE_K":np.sqrt(np.mean((y_fit-X_fit@bounded_fit.x)**2)),"active_bounds":str(bounded_fit.active_mask.tolist())}])
    display(fit_comparison)
    print("OLS lies inside bounds:",bool(((ols_coefficients>=LOWER_BOUNDS)&(ols_coefficients<=UPPER_BOUNDS)).all()))
    print("Coefficient difference:",bounded_fit.x-ols_coefficients)
    export_table(fit_comparison,"fit_comparison.csv")
    fit_display=fit_comparison.rename(columns={"method":"Method","c_K_per_kW":"c (K/kW)",
        "N_fit":"Fit N","deltaT_RMSE_K":"ΔT RMSE (K)","active_bounds":"Active bounds"})
    fig=table_figure(fit_display,[.14,.14,.16,.12,.20,.24],12)
    save_figure(fig,"F62-04",["data/reference_building/identification_data.csv"],"6–26 January 2025","K;K/kW","OLS vs bounded; b=[1e-5,.4]; c=[1e-5,.5]","Task B fit")
    fit_prediction=fit_transitions.indoor_temperature_c.to_numpy()+X_fit@bounded_fit.x
    fit_record=fit_transitions.copy()
    fit_record["actual_c"]=fit_record.next_temperature_c
    fit_record["prediction_c"]=fit_prediction
    fit_record["residual_c"]=fit_record.actual_c-fit_record.prediction_c
    export_table(fit_record,"fit_transitions.csv")
    ''')
    md('i3',r'''
    ## I3. Task C — effective parameters (slide 15; 8 min)

    Recover $C=dt/c$, $R=c/b$ and $\tau=dt/b$; also calculate $a=1-b$.
    These formulae assume forward Euler and **useful heat** as input.
    Record units and explain how omitted gains and sensor noise affect the estimates.

    Check your expressions with $b=0.025$, $c=0.125$ K/kW and $dt=0.5$ h:
    $C=4$ kWh/K, $R=5$ K/kW, $\tau=20$ h and $a=0.975$.
    ''')
    task('C','''
    estimated_C=dt/c
    estimated_R=c/b
    estimated_tau=dt/b
    a=1-b
    ''','''
    # TODO C: recover estimated_C, estimated_R, estimated_tau and a from b,c,dt.
    raise NotImplementedError("Task C: recover effective parameters with their units.")
    ''')
    code('parameter-results','''
    parameter_table=pd.DataFrame([{"b":b,"c_K_per_kW":c,"a":a,"dt_hours":dt,
        "R_K_per_kW":estimated_R,"C_kWh_per_K":estimated_C,"tau_hours":estimated_tau}])
    display(parameter_table);export_table(parameter_table,"student_model_parameters.csv")
    print("Separate hand example, not the fit: b=.025,c=.125,dt=.5 gives R=5,C=4,tau=20,a=.975.")
    ''')
    md('break',r'''
    Correlated inputs, heating control, measurement noise and omitted gains
    affect the estimated coefficients. Interpret the recovered $R,C$ as
    effective values for this model and fitting period.
    **Save and take the 15–20-minute break after slide 16.**
    ''')
    md('i4',r'''
    ## I4. Prediction protocols (slides 17–20)

    Freeze b,c. One-step prediction starts from measured $T_k$ at every interval.
    Free rollout starts once at the first validation observation and uses its own
    previous prediction thereafter. Both receive the recorded useful heat and
    outdoor inputs. The errors therefore describe performance conditional on
    these observed inputs. Operational forecasting also requires forecasts of
    future inputs.
    ''')
    code('protocol-diagram',r'''
    fig,axes=plt.subplots(2,1,figsize=(12,4.3))
    for ax,label in zip(axes,["One-step: measured reset each interval","Free rollout: one initial measurement"]):
        ax.set(xlim=(-.4,4.3),ylim=(-.5,1.4));ax.axis("off")
        ax.text(-.3,1.1,label,fontsize=14)
        for k in range(4):
            colour="black" if (ax is axes[0] or k==0) else "tab:orange"
            ax.plot(k,.25,"o",color=colour,ms=8)
            ax.text(k,-.25,f"t{k}",ha="center",fontsize=12)
            if k<3:ax.annotate("",xy=(k+1,.25),xytext=(k,.25),arrowprops={"arrowstyle":"->","color":"tab:blue"})
        ax.text(3.35,.3,"Common targets\nt1,t2,t3",fontsize=12)
    fig.suptitle("Initialisation at t0 is not a scored prediction",fontsize=14)
    save_figure(fig,"F62-05",units="time;°C",case="Different measured-state information sets")
    ''')
    md('task-d-text','''
    ### Task D — generate aligned predictions (slide 19; 12 min)

    Complete the one-step expression and a short rollout recurrence. Retain the
    initial point for plotting, but index scored predictions at the 335 target
    times only (27 January 00:30 to 2 February 23:30). At each subsequent step,
    use the previous predicted indoor temperature in the rollout recurrence.
    ''')
    task('D','''
    target_index=pd.DatetimeIndex(validation_transitions.target_time,name="target_time")
    one_step=pd.Series(validation_transitions.indoor_temperature_c.to_numpy()
        +b*validation_transitions.temperature_difference_c.to_numpy()
        +c*validation_transitions.delivered_heat_kw.to_numpy(),index=target_index,name="one_step_c")
    rollout_values=np.empty(len(validation_observations))
    rollout_values[0]=validation_observations.indoor_temperature_c.iloc[0]
    for k in range(len(validation_observations)-1):
        rollout_values[k+1]=rollout_values[k]+b*(validation_observations.outdoor_temperature_c.iloc[k]-rollout_values[k])+c*validation_observations.delivered_heat_kw.iloc[k]
    rollout_states=pd.Series(rollout_values,index=validation_observations.index,name="rollout_c")
    rollout=rollout_states.iloc[1:].rename_axis("target_time")
    actual=pd.Series(validation_transitions.next_temperature_c.to_numpy(),index=target_index,name="actual_c")
    ''','''
    target_index = pd.DatetimeIndex(validation_transitions.target_time, name="target_time")
    # TODO D: fill the one-step prediction from the measured current temperature.
    one_step = pd.Series(np.nan, index=target_index, name="one_step_c")
    rollout_values = np.empty(len(validation_observations))
    rollout_values[0] = validation_observations.indoor_temperature_c.iloc[0]
    for k in range(len(validation_observations) - 1):
        # TODO D: use rollout_values[k], outdoor temperature and useful heat.
        rollout_values[k + 1] = np.nan
    rollout_states = pd.Series(rollout_values, index=validation_observations.index, name="rollout_c")
    rollout = rollout_states.iloc[1:].rename_axis("target_time")
    actual = pd.Series(validation_transitions.next_temperature_c.to_numpy(), index=target_index, name="actual_c")
    raise NotImplementedError("Task D: align both predictions on targets, excluding initialisation from scores.")
    ''')
    code('validation-plots','''
    validation_record=validation_transitions.copy()
    validation_record["actual_c"]=actual.to_numpy()
    validation_record["one_step_c"]=one_step.to_numpy()
    validation_record["rollout_c"]=rollout.to_numpy()
    validation_record["one_step_residual_c"]=actual.to_numpy()-one_step.to_numpy()
    validation_record["rollout_residual_c"]=actual.to_numpy()-rollout.to_numpy()
    export_table(validation_record,"validation_transitions.csv")
    fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    axes[0].plot(validation_observations.index,validation_observations.indoor_temperature_c,color="black",label="Measured")
    axes[0].plot(one_step.index,one_step,color="tab:blue",label="One-step")
    axes[0].plot(rollout_states.index,rollout_states,color="tab:orange",label="Seven-day free rollout")
    axes[0].scatter([validation_observations.index[0]],[rollout_states.iloc[0]],color="0.5",marker="s",zorder=5,label="Initial state (unscored)")
    axes[0].set(ylabel="Indoor temperature (°C)",title="Validation: fixed model; 335 common scored targets")
    legend_above(axes[0],2);axes[0].set_title(axes[0].get_title(),pad=50)
    axes[1].plot(actual.index,actual-one_step,color="tab:blue",label="One-step residual")
    axes[1].plot(actual.index,actual-rollout,color="tab:orange",label="Rollout residual")
    axes[1].axhline(0,color="black",lw=1);legend_above(axes[1],2)
    axes[1].set(ylabel="Measured − predicted (K)",xlabel="Target time");time_axis(axes[1])
    axes[1].set_xlim(validation_observations.index[0],validation_observations.index[-1])
    save_figure(fig,"F62-06",["data/reference_building/identification_data.csv"],"27 January–2 February 2025","°C;K","One-step vs seven-day free rollout","Task D predictions")
    ''')
    md('i5',r'''
    ## I5. Task E — common-target scores (slide 21; 8 min)

    Use the supplied familiar metric helper with error = actual − prediction.
    Positive bias means underprediction. Complete the three score rows on the
    identical 335 targets. Persistence uses measured $T_k$ and is a **one-step**
    benchmark; the seven-day rollout uses only the initial indoor measurement.
    Compare the two one-step methods directly, then assess accumulated rollout
    drift separately. Report temperature errors in K.
    ''')
    code('metrics','''
    def temperature_metrics(actual,prediction):
        if not actual.index.equals(prediction.index):raise ValueError("Metric timestamps must match exactly.")
        y,p=actual.to_numpy(dtype=float),prediction.to_numpy(dtype=float)
        if y.shape!=p.shape or not np.isfinite(np.column_stack([y,p])).all():
            raise ValueError("Metrics require finite, equally sized pairs.")
        error=y-p
        return {"N":len(error),"MAE_K":float(np.mean(np.abs(error))),
                "RMSE_K":float(np.sqrt(np.mean(error**2))),"bias_K":float(np.mean(error))}
    ''')
    code('score-cases','''
    persistence=pd.Series(validation_transitions.indoor_temperature_c.to_numpy(),index=actual.index)
    prediction_cases={"One-step persistence":persistence,"One-step fitted":one_step,"Seven-day free rollout":rollout}
    information_sets={"One-step persistence":"Measured current T only",
        "One-step fitted":"Measured current T; recorded interval inputs",
        "Seven-day free rollout":"One initial T; recorded future inputs"}
    ''')
    task('E','''
    score_table=pd.DataFrame([{"method":name,"information_set":information_sets[name],
                              **temperature_metrics(actual,pred)} for name,pred in prediction_cases.items()])
    ''','''
    # TODO E: build score_table with method, information_set and temperature_metrics for each case.
    raise NotImplementedError("Task E: calculate MAE, RMSE and actual-minus-predicted bias on common targets.")
    ''')
    code('score-plot','''
    display(score_table);export_table(score_table,"validation_scores.csv")
    score_display=score_table.rename(columns={"method":"Method","information_set":"Information available",
        "MAE_K":"MAE (K)","RMSE_K":"RMSE (K)","bias_K":"Bias (K)"})
    fig=table_figure(score_display,[.21,.32,.08,.13,.13,.13],11,4.2)
    save_figure(fig,"F62-07",["data/reference_building/identification_data.csv"],"335 targets in 27 January–2 February 2025","K","Different information sets; no common forecast ranking","Task E metrics")
    ''')
    md('i6',r'''
    ## I6. Residual diagnostics (slides 23–24)

    A residual is the observed temperature minus its prediction:
    $e_{k+1}=T_{k+1}-\hat T_{k+1}$. Positive residuals indicate underprediction.
    Pair each residual with the inputs during the preceding interval. For
    example, the error at 10:30 corresponds to the heating input from
    10:00 to 10:30 and the outdoor temperature recorded at 10:00.

    Inspect the residual time series, its association with heating power and
    its distribution by hour. Which pattern suggests an omitted heat gain or
    additional thermal dynamics? These plots describe associations; causal
    interpretation requires further evidence.
    ''')
    code('residuals','''
    residual_records=validation_record[["predictor_time","target_time","delivered_heat_kw","outdoor_temperature_c","one_step_residual_c"]].copy()
    residual_records["target_hour"]=residual_records.target_time.dt.hour
    export_table(residual_records,"residual_diagnostics.csv")
    fig,(ax0,ax1,ax2)=plt.subplots(1,3,figsize=(12,4.2),sharey=True)
    ax0.plot(residual_records.target_time,residual_records.one_step_residual_c,color="tab:blue")
    ax0.set(ylabel="One-step residual (K)",xlabel="Temperature observation time",title="Residual over time")
    ax0.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax0.set_xlim(actual.index[0],actual.index[-1])
    ax1.scatter(residual_records.delivered_heat_kw,residual_records.one_step_residual_c,s=12,alpha=.55,color="tab:blue")
    ax1.set(xlabel="Heating input during interval k (kW)",title="Residual versus heating input")
    ax2.boxplot([residual_records.loc[residual_records.target_hour.eq(h),"one_step_residual_c"] for h in range(24)],positions=range(24),showfliers=True)
    ax2.set(xticks=range(0,24,3),xticklabels=range(0,24,3),xlabel="Hour of temperature observation",title="Residual by time of day")
    for ax in [ax0,ax1,ax2]:ax.axhline(0,color="black",lw=1,linestyle="--")
    save_figure(fig,"F62-08",["data/reference_building/identification_data.csv"],"Validation period","K;kW;hour","One-step residual; heating input during the preceding interval")
    ''')
    md('i7',r'''
    ## I7. Real-house experiment: November and June (slides 25–31)

    We now fit a 1R1C model to measurements from an occupied house. Real
    measurements vary with weather and occupancy, and the available power
    signal approximates the heat entering the building. A single indoor temperature simplifies
    the thermal behaviour of several rooms and the building fabric. We will
    inspect these limitations and examine how the training period affects the
    fitted coefficients and predictions.

    The local files contain 11–15 November 2016 and 10–14 June 2017, each with
    720 observations at 10-minute intervals. See the
    [data README](data/tutorial_6_2/README.md) and
    [Hollick and Wingfield (2018)](https://doi.org/10.14324/000.ds.10087216).

    1. Inspect the temperature, power and irradiance signals.
    2. Fit the same model to November and the first three days of June.
    3. Compare the coefficients and effective parameters.
    4. Test both models on the final two days of June.

    The June test observations are held out from both fits. November contributes
    five fitting days and June contributes three; report the training row counts
    alongside the coefficients. The comparison describes sensitivity to these
    training datasets, including their different lengths and operating conditions.

    This section can also run independently: execute setup, supplied plotting
    helpers, the metric-function cell, and I7. The student tasks in I7 use only
    the house measurements.
    ''')
    code('house-data','''
    house_raw = {}
    for month, filename in [("November", "10mins_solpap_2016.csv"),
                            ("June", "10mins_solpap_2017.csv")]:
        raw = pd.read_csv(ROOT / "data/tutorial_6_2" / filename, index_col=0)
        raw.index = pd.to_datetime(raw.index, format="%d/%m/%Y %H:%M")
        house_raw[month] = raw

    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex="col", sharey="row")
    for col, (month, raw) in enumerate(house_raw.items()):
        axes[0, col].plot(raw.index, raw["T_Average (degC)"], label="Indoor", color="black")
        axes[0, col].plot(raw.index, raw["T_External (degC)"], label="Outdoor", color="0.5")
        axes[0, col].set_title(month)
        axes[0, col].legend()
        axes[1, col].plot(raw.index, raw["P_tot (W)"] / 1000, color="tab:orange")
        axes[2, col].plot(raw.index, raw["Solar (W/m2)"] / 1000, color="tab:green")
        axes[2, col].set_xlabel("Observation time")
        time_axis(axes[2, col])
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[1, 0].set_ylabel("Electricity + gas (kW)")
    axes[2, 0].set_ylabel("Irradiance (kW/m²)")
    save_figure(fig, "F62-12", ["data/tutorial_6_2/10mins_solpap_2016.csv",
        "data/tutorial_6_2/10mins_solpap_2017.csv"], "November 2016; June 2017",
        "°C;kW;kW/m²", "Raw measurements; gaps retained")
    ''')
    md('house-inspect',r'''
    Compare heating power and irradiance between the two periods. Why might
    coefficients fitted during the heating season predict summer temperatures
    poorly? Which physical processes would a one-state model approximate?

    Fill missing readings using linear interpolation within each fitting and
    test period, as supplied below. We use these interpolated measurements
    for this offline identification and prediction comparison.
    ''')
    md('house-math',r'''
    ### 1R1C model with effective solar gains

    We represent the building with indoor temperature $T$ and outdoor
    temperature $T_o$, both in °C. Let $P$ be electricity-plus-gas power in kW
    and $S$ be measured solar irradiance in kW/m². We approximate the heat
    balance by

    $$C_{\mathrm{eff}}\frac{dT}{dt}
    =\frac{T_o-T}{R_{\mathrm{eff}}}+P+g_{\mathrm{eff}}S.$$

    Here $R_{\mathrm{eff}}$ is in K/kW and $C_{\mathrm{eff}}$ is in kWh/K.
    The effective solar-gain factor $g_{\mathrm{eff}}$ has units of m², so
    $g_{\mathrm{eff}}S$ is a heat gain in kW. It combines the effects of window
    area, transmission, orientation and shading relative to the measured
    irradiance. Total energy consumption serves as a heat-input proxy; its
    conversion to useful indoor heat is absorbed into the effective parameters.

    With $\Delta t=1/6$ h, forward Euler gives

    $$\Delta T_k=b(T_{o,k}-T_k)+cP_k+dS_k+\epsilon_k,$$

    $$b=\frac{\Delta t}{R_{\mathrm{eff}}C_{\mathrm{eff}}},\qquad
    c=\frac{\Delta t}{C_{\mathrm{eff}}},\qquad
    d=\frac{\Delta t\,g_{\mathrm{eff}}}{C_{\mathrm{eff}}}.$$

    The residual $\epsilon_k$ includes sensor error, unmeasured gains and
    thermal dynamics omitted by the 1R1C approximation. We use the same three
    predictors in each period:

    $$x_k=[T_{o,k}-T_k,\ P_k,\ S_k],\qquad
    y_k=T_{k+1}-T_k,\qquad \theta=[b,c,d]^T.$$

    For $N$ complete transitions, $X$ has shape $(N,3)$ and $y$ has shape
    $(N,)$. The model has its intercept fixed at zero. For each fitting period,
    estimate $\theta$ by minimising $\|X\theta-y\|_2^2$ with the supplied bounds.
    ''')
    md('r1',r'''
    ### Task R1. Prepare three predictors (10 min)

    Complete `make_house_transitions(observations)`. Return a DataFrame with
    measured `T`, `T_next`, target timestamps, `T_diff`, power `P_kW` and
    irradiance `S_kW_m2`. Convert both power and irradiance from W to kW units.
    Calculate `dT`, then drop the final row, whose next temperature is missing.

    Check one row by hand: if $T_k=20$°C, $T_{k+1}=20.1$°C, $T_{o,k}=10$°C,
    $P_k=2000$ W and $S_k=500$ W/m², the feature row is $[-10,2,0.5]$ and
    its target is $0.1$ K. The supplied next cell calls your function separately
    within November fit, June fit and June test observations.
    ''')
    task('R1','''
    def make_house_transitions(observations):
        T = observations["T_Average (degC)"]
        table = pd.DataFrame({
            "T": T,
            "T_next": T.shift(-1),
            "target_time": observations.index.to_series().shift(-1),
            "T_diff": observations["T_External (degC)"] - T,
            "P_kW": observations["P_tot (W)"] / 1000,
            "S_kW_m2": observations["Solar (W/m2)"] / 1000,
        })
        table["dT"] = table["T_next"] - table["T"]
        return table.dropna()
    ''','''
    def make_house_transitions(observations):
        T = observations["T_Average (degC)"]
        # TODO R1: fill T_diff, P_kW and S_kW_m2 in the feature table.
        table = pd.DataFrame({
            "T": T,
            "T_next": T.shift(-1),
            "target_time": observations.index.to_series().shift(-1),
            "T_diff": np.nan,
            "P_kW": np.nan,
            "S_kW_m2": np.nan,
        })
        # TODO R1: add dT and return table.dropna() to remove the final row.
        raise NotImplementedError("Task R1: return the 10-minute feature and target table.")
    ''')
    code('house-split','''
    june_split = pd.Timestamp("2017-06-13")
    # Fill each period separately with linear interpolation.
    november_observations = house_raw["November"].interpolate(method="linear")
    june_fit_observations = house_raw["June"].loc[house_raw["June"].index < june_split].interpolate(method="linear")
    june_test_observations = house_raw["June"].loc[house_raw["June"].index >= june_split].interpolate(method="linear")

    november_fit = make_house_transitions(november_observations)
    june_fit = make_house_transitions(june_fit_observations)
    june_test = make_house_transitions(june_test_observations)
    house_features = ["T_diff", "P_kW", "S_kW_m2"]

    house_periods = {"November fit": november_fit, "June fit": june_fit, "June test": june_test}
    house_counts = pd.DataFrame([
        {"period": name, "N_transitions": len(frame),
         "first_predictor": frame.index.min(), "last_target": frame.target_time.max()}
        for name, frame in house_periods.items()
    ])
    display(november_fit.head(3))
    display(house_counts)
    export_table(house_counts, "house_transition_counts.csv")
    ''')
    md('r2',r'''
    ### Task R2. Fit November and June models (15 min)

    Fit $b,c,d$ twice with `lsq_linear`: once using `november_fit`, then using
    `june_fit`. Use `house_features` in the supplied order and `dT` as the
    target. Store the results as `november_model` and `june_model`.
    Check solver success and compare their coefficient arrays in the next cell.

    The shared bounds are $b\in[10^{-6},0.2]$, $c\in[10^{-6},0.5]$ K/kW
    and $d\in[0,2]$ K·m²/kW. They specify broad allowed ranges at the
    10-minute step. The positive lower bounds on $b,c$ allow the following
    parameter calculations; estimates at a bound require careful interpretation.

    Recover the effective parameters using

    $$R_{\mathrm{eff}}=\frac{c}{b},\qquad
    C_{\mathrm{eff}}=\frac{\Delta t}{c},\qquad
    \tau=\frac{\Delta t}{b},\qquad
    g_{\mathrm{eff}}=\frac{d}{c}.$$

    For example, $b=0.01$, $c=0.05$ K/kW and $d=0.1$ K·m²/kW give
    $g_{\mathrm{eff}}=2$ m² and a solar heat gain of 1 kW at $S=0.5$ kW/m².
    ''')
    code('house-bounds','''
    house_dt = 1 / 6
    house_lower = np.array([1e-6, 1e-6, 0.0])
    house_upper = np.array([0.2, 0.5, 2.0])
    ''')
    task('R2','''
    november_model = lsq_linear(november_fit[house_features], november_fit["dT"],
                                bounds=(house_lower, house_upper))
    june_model = lsq_linear(june_fit[house_features], june_fit["dT"],
                            bounds=(house_lower, house_upper))
    if not november_model.success or not june_model.success:
        raise RuntimeError("Inspect the least-squares solver messages.")
    november_coefficients = november_model.x
    june_coefficients = june_model.x
    ''','''
    # TODO R2: fit each training table with the same three predictors and bounds.
    # Store november_model, june_model and their .x coefficient arrays as
    # november_coefficients and june_coefficients.
    raise NotImplementedError("Task R2: fit separate November and June coefficients.")
    ''')
    code('house-coefficients','''
    coefficient_table = pd.DataFrame(
        [november_coefficients, june_coefficients],
        index=["November", "June"],
        columns=["b", "c_K_per_kW", "d_K_m2_per_kW"])
    parameter_rows = []
    for month, model, n_fit in [("November", november_model, len(november_fit)),
                               ("June", june_model, len(june_fit))]:
        b_house, c_house, d_house = model.x
        parameter_rows.append({
            "fit_period": month, "N_fit": n_fit,
            "R_eff_K_per_kW": c_house / b_house,
            "C_eff_kWh_per_K": house_dt / c_house,
            "tau_hours": house_dt / b_house,
            "g_eff_m2": d_house / c_house,
            "active_bounds": model.active_mask.tolist(),
        })
    house_parameters = pd.DataFrame(parameter_rows)
    display(coefficient_table)
    display(house_parameters)
    export_table(coefficient_table, "house_seasonal_coefficients.csv", index=True)
    export_table(house_parameters, "house_effective_parameters.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, column, units in zip(axes, coefficient_table.columns,
                                ["dimensionless", "K/kW", "K·m²/kW"]):
        ax.bar(coefficient_table.index, coefficient_table[column], color=["tab:blue", "tab:orange"])
        ax.set(ylabel=units, title=column.split("_")[0])
    save_figure(fig, "F62-11", ["data/tutorial_6_2/10mins_solpap_2016.csv",
        "data/tutorial_6_2/10mins_solpap_2017.csv"], "November; first three days of June",
        "dimensionless;K/kW;K·m²/kW", "Same model and coefficient bounds; linearly interpolated measurements")
    ''')
    md('house-coefficient-discussion',r'''
    Which coefficients change most between the two fits? Compare the input
    ranges in the raw plots and check `active_bounds` (−1 means lower bound,
    0 means interior, +1 means upper bound). Explain how power, irradiance,
    occupant behaviour and omitted thermal states can affect the estimates.

    Both fits describe the same house using different observations. Each fit
    approximates the measured temperature changes with three terms. Its
    coefficients can absorb effects of omitted processes and correlated inputs.
    A change in fitted $R_{\mathrm{eff}}$, $C_{\mathrm{eff}}$ or
    $g_{\mathrm{eff}}$ alone is insufficient evidence of a change in the fabric
    or window properties.
    ''')
    md('r3',r'''
    ### Task R3. Compare transfer and refitting on June (15 min)

    For every row in `june_test`, predict the next temperature twice:

    $$\hat T_{k+1}^{\mathrm{Nov}}=T_k+x_k\hat\theta_{\mathrm{Nov}},\qquad
    \hat T_{k+1}^{\mathrm{Jun}}=T_k+x_k\hat\theta_{\mathrm{Jun}}.$$

    Complete the two prediction columns in `june_predictions`, then calculate
    MAE, RMSE and bias with `temperature_metrics`. Keep the supplied
    persistence prediction $\hat T_{k+1}=T_k$ as a reference. All three use
    the same current temperature and the same 10-minute test targets.
    The RC models also use recorded interval power and irradiance.

    The November coefficients remain fixed during June evaluation. The June
    model uses only 10–12 June for fitting; 13–14 June supplies the test targets.
    Use `june_test` as the common index while calculating metrics. The supplied
    plot places each prediction at its `target_time`.
    ''')
    task('R3','''
    june_predictions = pd.DataFrame(index=june_test.index)
    june_predictions["November transfer"] = june_test["T"] + june_test[house_features] @ november_coefficients
    june_predictions["June refit"] = june_test["T"] + june_test[house_features] @ june_coefficients
    june_predictions["Persistence"] = june_test["T"]
    june_scores = pd.DataFrame([
        {"model": name, **temperature_metrics(june_test["T_next"], prediction)}
        for name, prediction in june_predictions.items()
    ])
    ''','''
    june_predictions = pd.DataFrame(index=june_test.index)
    # TODO R3: use T + X @ coefficients for each of the two fitted models.
    june_predictions["November transfer"] = np.nan
    june_predictions["June refit"] = np.nan
    june_predictions["Persistence"] = june_test["T"]
    # TODO R3: build june_scores with model, N, MAE_K, RMSE_K and bias_K.
    raise NotImplementedError("Task R3: compare both models on the same June test targets.")
    ''')
    code('house-transfer-figure','''
    display(june_scores)
    export_table(june_scores, "house_june_test_scores.csv")
    june_record = june_test[["target_time", "T", "T_next"]].join(june_predictions)
    export_table(june_record.rename_axis("predictor_time"), "house_june_test_predictions.csv", index=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(june_test.target_time, june_test["T_next"], color="black", label="Measured")
    for name, colour in [("November transfer", "tab:blue"), ("June refit", "tab:orange")]:
        axes[0].plot(june_test.target_time, june_predictions[name], color=colour, alpha=.8, label=name)
        axes[1].plot(june_test.target_time, june_test["T_next"] - june_predictions[name], color=colour, label=name)
    axes[0].set(ylabel="Indoor temperature (°C)", title="June hold-out: 10-minute one-step predictions")
    axes[0].legend(ncol=3)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="Measured − predicted (K)", xlabel="Target time")
    axes[1].legend(ncol=2)
    axes[1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\\n%H:%M"))
    save_figure(fig, "F62-10", ["data/tutorial_6_2/10mins_solpap_2016.csv",
        "data/tutorial_6_2/10mins_solpap_2017.csv"], "13–14 June 2017",
        "°C;K", "Fixed November fit versus June refit; interpolated June test measurements",
        counts=f"{len(june_test)} common 10-minute targets per model")
    ''')
    md('house-interpret',r'''
    Use your computed coefficients, scores and residual plots to write four
    sentences: identify one limitation of the heat-input proxy; describe one simplifying
    model assumption; quantify one coefficient difference; and state how
    transfer and refitting compare on the June test set.

    Does either fitted model improve on persistence? A 10-minute one-step
    prediction starts from the latest indoor measurement, so small errors
    alone provide limited evidence about longer simulations. Refitting can
    improve performance when the new training period represents the test
    conditions; the computed scores determine the outcome here.

    The 1R1C model approximates the occupied house. Heat-input proxies,
    measurement noise and omitted processes affect both parameter estimates
    and prediction errors. Training-period sensitivity is therefore part of
    the result. Name an additional measurement or model term that could help,
    and specify a new period on which you would evaluate that revision.
    ''')
    md('conclusion',r'''
    ## Conclusion

    A 1R1C model represents an occupied house with a simplified heat balance
    and an approximate heat-input signal. Its fitted coefficients summarise the selected
    observations under these assumptions. Comparing November and
    June exposes their sensitivity to the training data. Evaluate transfer
    and refitting on shared held-out targets, and interpret the effective
    parameters together with the prediction errors and model assumptions.
    ''')
    code('manifest','''
    if EXPORT_ARTEFACTS:
        pd.DataFrame(figure_manifest).sort_values("id").to_csv(ROOT/"teaching/week6_day2_figure_manifest.csv",index=False)
    ''')
    md('references','''
    ### References and source credits

    - Synthetic data assumptions: [reference data README](data/reference_building/README.md).
    - House measurements: [Hollick and Wingfield (2018), UCL dataset](https://discovery.ucl.ac.uk/id/eprint/10087216/),
      DOI 10.14324/000.ds.10087216; [local data README](data/tutorial_6_2/README.md).
    - Teaching reference: [BENV0092 Exercise 3 solution](https://github.com/akylasstrat/BENV0092/blob/main/exercise_3_RC_model_solution.ipynb).
      I7 uses its three-predictor formulation with explicit units and a shared June hold-out.
    - [SciPy 1.15.3 lsq_linear](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.optimize.lsq_linear.html).
    - Bacher and Madsen (2011), Energy and Buildings 43, 1511–1522;
      Drgoňa et al. (2020), Annual Reviews in Control 50, 190–232.
    ''')
    return nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})


if __name__=='__main__':
    for solution in [False,True]:nbf.write(make_notebook(solution),ROOT/(STEM+('_solution' if solution else '')+'.ipynb'))
    with (ROOT/'teaching/week6_day2_task_manifest.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['task_id','section','slide','minutes_including_debrief','cell_id'])
        for letter,section,slide,minutes in TASKS:writer.writerow([letter,section,slide,minutes,'w62-task-'+letter.lower()])
    with (ROOT/'teaching/week6_day2_schedule.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['slide','core_minutes','block'])
        for slide,minutes in enumerate(SLIDE_MINUTES,1):
            block=next(i for i,(first,last) in enumerate(TEACHING_BLOCKS,1) if first<=slide<=last)
            writer.writerow([slide,minutes,block])
    print('Built Tutorial 6.2 pair only; Tasks A–E and main real-house Tasks R1–R3.')
