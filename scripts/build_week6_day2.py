"""Scoped Tutorial 6.2 authoring. Reconcile live manual edits before rebuilding."""
from pathlib import Path
import csv
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STEM = 'tutorial_6_2_rc_model_identification'
TASKS = [('A','I1',7,12),('B','I2',13,12),('C','I3',15,8),
         ('D','I4',19,12),('E','I5',21,8),('F','I6',25,12)]
SLIDE_MINUTES = [3,3,6,5,6,5,12,5,5,6,5,5,12,6,8,8,5,5,12,6,8,4,5,5,12,6,8,8,6,5,5]
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
 ('F62-09','w6d2_F09_horizon_validation','26','I6','6 origins; 48 leads; 288 origin-target pairs','Left: RMSE at each lead across six origins. Blue uses updated measured states for repeated one-step predictions, not a 24 h forecast issued at midnight; orange initialises only at midnight. Both agree at 0.5 h. Right: rollout RMSE over all targets in each first 6/12/24 h window. Recorded future inputs condition both protocols.'),
 ('F62-10','w6d2_F10_measured_house_transfer','28','I7','720 observations per period; valid counts computed','Ten-minute one-step predictions and segmented rollouts with a frozen November fit. November scores are in-sample; June uses unchanged coefficients. Markers identify restarts after missing-input gaps, not an uninterrupted five-day forecast. Segment lengths differ, so this is not a common-horizon comparison. Electricity-plus-gas power is a useful-heat proxy; parameters are effective.'),
 ('F62-11','w6d2_F11_fit_period_sensitivity','35 (optional)','E3','4 fitting windows; valid counts computed','Same predictors, bounds and original-observation masks across estimation windows. Effective parameters may vary with input variation, correlations, omitted gains and noise; this does not establish changing fabric. June refit (post-hoc) is distinct from frozen-model June evaluation; bars are estimates, not confidence intervals.'),
]


def make_notebook(solution=False):
    cells=[]
    def md(key,source):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(),id='w62-'+key))
    def code(key,source,tags=None):cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(),id='w62-'+key,metadata={'tags':tags or ['supplied']}))
    def task(letter,answer,scaffold,optional=False):
        code('task-'+letter.lower(),answer if solution else scaffold,['optional_task' if optional else 'task',letter])

    md('purpose',r'''
    # Tutorial 6.2: RC model identification and validation

    In this tutorial, we fit effective thermal parameters, then test one-step and free
    rollout behaviour on later observations. A supplied real-house demonstration
    and optional exercises provide a case with real-world measurement data.

    **Outcomes:** align transitions; fit bounded coefficients; recover effective
    $R,C$ and time constant parameter values; compare common-target errors; diagnose horizon limits.

    | Block | Slides | Minutes | Focus |
    |---|---|---:|---|
    | 1 | 1–9 | 50 | Measurements, timing and Task A |
    | 2 | 10–16 | 50 | Estimation and Tasks B–C |
    | Break | after 16 | 15–20 additional | Save and pause |
    | 3 | 17–24 | 50 | Validation and Tasks D–E |
    | 4 | 25–31 | 50 | Task F, real-data transfer and handover |

    The six tasks total 64 minutes within 200 teaching minutes. Optional E1–E4
    take about 35–45 minutes outside the core. All plots are supplied. Run from
    the repository root; no earlier tutorial, saved predictions or download is needed.
    ''')
    md('i0',r'''
    ## I0. Data, units and purpose (slides 3–4)

    The frozen [reference data](data/reference_building/README.md) contain 28 days
    of noisy synthetic indoor measurements at 30-minute intervals. The row at
    $t_k$ contains measured $T_k$, outdoor $T_{o,k}$ and useful heat $q_{th,k}$;
    the latter inputs drive $T_k\rightarrow T_{k+1}$. The last row has no observed
    next target. The separate fabric state is not an identification input.

    Fit $\Delta T_k=b(T_{o,k}-T_k)+cq_{th,k}+\epsilon_k$, without an intercept.
    Use kW, hours, K/kW, kWh/K and °C. Solar/internal gains and extra plant states
    are omitted. Noise in $T_k$ enters both the predictor and differenced target:
    bounded least squares is not an unbiased physical-parameter estimator.
    The supplied generator uses sensor noise σ=0.06°C and process noise
    σ=0.012°C per air-state interval (one third for fabric). These are simulator
    settings, not fitted error variances; the heat signal also responds to temperature limits.

    We keep the synthetic core for a controlled timing example. The real-house
    measurements are already supplied locally in `data/tutorial_6_2`; see the
    [local data README](data/tutorial_6_2/README.md) and
    [Hollick and Wingfield (2018), dataset DOI](https://doi.org/10.14324/000.ds.10087216).
    This notebook provides the complete transfer demonstration before optional work.
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
    # TODO A: form next-time/temperature, changes and differences WITHIN each period.
    # Keep finite, consecutive transitions; build X_fit, y_fit and both transition tables.
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
    Bounds $b\in[10^{-5},0.4]$ and $c\in[10^{-5},0.5]$ are broad teaching
    assumptions at dt=0.5 h; c has K/kW units. They are not confidence intervals.
    $0<b\leq1$ gives non-oscillatory Euler relaxation; stability permits $0<b<2$.

    Complete `lsq_linear`, check success and obtain b,c. Do not change the bounds
    or inputs to force a contrast with OLS.
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
    These formulae assume Euler coefficients and **useful heat** as input.
    They do not directly invert exact-discrete or electricity-proxy coefficients.
    Record units and explain why these are effective model parameters.
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
    A plausible fit does not establish unique fabric properties. Correlated
    inputs, closed-loop operation, measurement noise and omitted gains can alter
    the coefficients. **Save and take the 15–20-minute break after slide 16.**
    ''')
    md('i4',r'''
    ## I4. Prediction protocols (slides 17–20)

    Freeze b,c. One-step prediction starts from measured $T_k$ at every interval.
    Free rollout starts once at the first validation observation and uses its own
    previous prediction thereafter. Both receive the recorded useful heat and
    outdoor inputs: this is **ex-post conditional validation**, not an operational
    forecast with genuinely issued future-weather predictions.
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
    times only (27 January 00:30 to 2 February 23:30). Do not reset the rollout
    from any later measured indoor state.
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
    # TODO D: predict from measured T for one-step; initialise once and recur for rollout.
    # Assign one_step, rollout and actual to target_index; keep rollout_states for plotting.
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
    benchmark; the seven-day rollout has a different information set. Do not rank
    these as equivalent forecasting methods. MAPE is unsuitable for Celsius temperatures.
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
    ## I6. Residuals and horizon evidence (slides 23–26)

    Pair $e_{k+1}=T_{k+1}-\hat T_{k+1}$ with heating input during interval $k$
    and outdoor temperature at predictor time: $q_{th,k},T_{o,k}$. For
    10:00→10:30, compare the temperature error at 10:30 with the inputs for
    10:00→10:30, not the inputs for the next interval. Positive residuals mean
    underprediction. These associations suggest model diagnostics; they do not
    prove that a plotted input caused the error. Keep both timestamps.

    An **origin** is the prediction start time; **lead** is elapsed time from
    origin to target; **horizon** is the prediction-window length evaluated.
    **Initialisation** sets the predicted indoor state to its measured value
    at the origin; it does not refit coefficients. Start once per day at midnight
    on 27 January–1 February. At each origin, one 48-step rollout supplies the
    first 12, 24 and 48 targets for 6, 12 and 24 h windows, with no extra reset
    at 6 or 12 h. All origins use the same fitted coefficients.

    Recorded later outdoor temperature and useful heat are ex-post conditioning
    inputs, not operational forecasts. Later indoor measurements enter only the
    explicitly labelled repeated one-step benchmark, never the free rollout.
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
    code('horizon-experiment','''
    # Origins depend on available complete 24 h windows, never on observed errors.
    midnight=validation_observations.index[(validation_observations.index.hour==0)&(validation_observations.index.minute==0)]
    origins=[t for t in midnight if t+pd.Timedelta(hours=24)<=validation_observations.index[-1]]
    horizon_rows=[]
    for origin in origins:
        position=validation_observations.index.get_loc(origin)
        predicted=float(validation_observations.indoor_temperature_c.loc[origin])
        for lead in range(1,49):
            row=validation_observations.iloc[position+lead-1]
            predicted=predicted+b*(row.outdoor_temperature_c-predicted)+c*row.delivered_heat_kw
            # Separate benchmark: a new measured-state prediction at every interval.
            measured_T_k=row.indoor_temperature_c
            one_step_prediction=measured_T_k+b*(row.outdoor_temperature_c-measured_T_k)+c*row.delivered_heat_kw
            target_time=validation_observations.index[position+lead]
            observed=validation_observations.indoor_temperature_c.loc[target_time]
            horizon_rows.append({"origin":origin,"lead_steps":lead,"lead_hours":lead*dt,
                "predictor_time":row.name,"target_time":target_time,"actual_c":observed,
                "prediction_c":predicted,"error_c":observed-predicted,
                "one_step_prediction_c":one_step_prediction,"one_step_error_c":observed-one_step_prediction})
    horizon_records=pd.DataFrame(horizon_rows)
    export_table(horizon_records,"multi_origin_records.csv")
    # Illustration only: three nested windows from the first midnight origin.
    first_origin_windows=pd.DataFrame({"start":[origins[0]]*3,"horizon_hours":[6,12,24],
        "end":[origins[0]+pd.Timedelta(hours=h) for h in [6,12,24]],"N_targets":[12,24,48]})
    display(first_origin_windows)
    ''')
    md('task-f-text','''
    ### Task F — aggregate horizon error and judge adequacy (slide 25; 12 min)

    The supplied loop uses six complete midnight origins, 27 January–1 February.
    Complete RMSE by lead from `horizon_records.error_c`. Keep all six origins
    for every lead: this is lead-specific RMSE across origins. The supplied next
    cell also reports window RMSE over all targets in each origin's first H hours,
    rather than only its endpoint. Use the result to state a useful horizon, one limitation and
    one next check; no universal acceptable RMSE threshold is supplied.
    ''')
    task('F','''
    lead_rmse=horizon_records.groupby("lead_hours").error_c.apply(lambda e:np.sqrt(np.mean(e**2))).rename("RMSE_K")
    ''','''
    # TODO F: group error_c by lead_hours and calculate RMSE for each lead.
    raise NotImplementedError("Task F: aggregate errors over the same six origins at every lead.")
    ''')
    code('horizon-plot','''
    window_rows=[]
    for origin,group in horizon_records.groupby("origin"):
        for hours in [6,12,24]:
            window=group.loc[group.lead_hours<=hours]
            window_rows.append({"origin":origin,"horizon_hours":hours,"N":len(window),
                                "RMSE_K":np.sqrt(np.mean(window.error_c**2))})
    window_scores=pd.DataFrame(window_rows)
    lead_summary=pd.concat([lead_rmse,horizon_records.groupby("lead_hours").size().rename("N_origins")],axis=1)
    # Apply the completed Task F pattern to the separate measured-state benchmark.
    one_step_lead_rmse=horizon_records.groupby("lead_hours").one_step_error_c.apply(lambda e:np.sqrt(np.mean(e**2))).rename("one_step_RMSE_K")
    benchmark_lead_summary=pd.concat([lead_summary,one_step_lead_rmse],axis=1)
    display(lead_summary.iloc[[0,11,23,47]]);display(window_scores)
    export_table(lead_summary,"lead_scores.csv",index=True);export_table(window_scores,"window_scores.csv")
    export_table(benchmark_lead_summary,"one_step_benchmark_lead_scores.csv",index=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    axes[0].plot(one_step_lead_rmse.index,one_step_lead_rmse,color=COLOURS["one_step"],label="One-step: updated measured state")
    axes[0].plot(lead_rmse.index,lead_rmse,color=COLOURS["rollout"],label="Rollout: midnight initialisation")
    axes[0].set(xlabel="Hours since midnight start",ylabel="RMSE (K)",xlim=(.5,24),title="One-step and rollout error by lead")
    legend_above(axes[0],1);axes[0].set_title(axes[0].get_title(),pad=65)
    for hours,group in window_scores.groupby("horizon_hours"):
        axes[1].plot(group.origin,group.RMSE_K,"o-",label=f"{hours} h / {int(hours/dt)} targets")
    axes[1].set(ylabel="RMSE over first H hours (K)",xlabel="Prediction start (midnight)",title="Rollout error over each prediction window")
    legend_above(axes[1],1);axes[1].set_title(axes[1].get_title(),pad=65);time_axis(axes[1])
    save_figure(fig,"F62-09",["data/reference_building/identification_data.csv"],"Origins 27 January–1 February 2025","K;h","Fixed fitted model; recorded future inputs","Task F lead aggregation")
    ''')
    md('horizon-discussion','''
    **Interpret:** distinguish the seven-day drift diagnostic from the shorter
    horizon tests. In F62-09, blue comprises repeated one-step predictions using
    updated measured states, not a 24 h forecast issued at midnight. Orange uses
    only the midnight indoor measurement. Both agree at lead 0.5 h because the
    initial state and first inputs coincide. Neither error curve must increase
    monotonically. The left panel pools six origins at each lead; the right
    scores every target within each origin's first H hours, not its endpoint alone.
    The 6/12/24 h windows reuse targets within each origin, so
    horizon scores are dependent. The selected midnight 24 h windows do not
    overlap one another's target sets, but consecutive days need not be independent.
    Any model revision based on these results needs new final evaluation data.
    ''')
    md('i7',r'''
    ## I7. Supplied real-house transfer (slides 27–28; 8 min each)

    These are occupied-house measurements: 11–15 November 2016 and 10–14 June
    2017, 720 observations each at 10 minutes. Source:
    [Hollick and Wingfield (2018), UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10087216/).
    The files are provided in `data/tutorial_6_2`; see the
    [local data README](data/tutorial_6_2/README.md). The one-step horizon is
    **10 minutes**. November is the fitting period, so its scores are in-sample
    diagnostics; June uses unchanged November coefficients before any June refit.

    Use $\Delta T=b(T_o-T)+cP^{proxy}+gS^{proxy}$ with no intercept.
    Convert `P_tot` W→kW and solar W/m²→kW/m². Electricity plus gas power is
    **not delivered useful space-heating output**. R,C recovered under that proxy
    assumption are effective, not identified fabric properties.

    Fit on originally observed November transitions only. Apply the fixed model
    to June before any June refit. Main scores use originally observed current
    and target temperatures and finite original forcing inputs. A missing forcing
    interval ends a rollout segment; the next supported segment starts explicitly
    from a measured temperature. Missing temperature targets alone do not reset it.
    These are segmented rollouts, not one uninterrupted five-day forecast.
    Markers identify each restart. Accurate one-step predictions can coexist
    with rollout drift, but segment lengths differ: this plot does not directly
    establish a common 6/12/24 h horizon effect. Use I6 for that comparison.
    No interpolation is used in the headline example. Optional E1 tests a limited
    input-imputation assumption separately. The demonstration needs no optional TODO.
    To explore real data alone, run setup, plotting helpers, the metric-function
    cell and this section; the synthetic fitting tasks are not dependencies.

    The data, estimation period and model assumptions affect effective R,C.
    Optional E3 compares fitting windows without changing this frozen-model
    evaluation; it does not by itself establish changing physical fabric.
    ''')
    code('house-data','''
    HOUSE_COLUMNS=["T_Average (degC)","T_External (degC)","P_tot (W)","Solar (W/m2)"]
    house_raw={}
    house_original_masks={}
    house_audit_rows=[]
    for name,filename in [("November","10mins_solpap_2016.csv"),("June","10mins_solpap_2017.csv")]:
        raw=pd.read_csv(ROOT/"data/tutorial_6_2"/filename,index_col=0)
        raw.index=pd.to_datetime(raw.index,format="%d/%m/%Y %H:%M")
        if raw.index.has_duplicates or not raw.index.equals(pd.date_range(raw.index[0],periods=len(raw),freq="10min")):
            raise ValueError("Inspect irregular or duplicate measured-house timestamps before continuing.")
        house_raw[name]=raw.copy(deep=True)
        mask=pd.DataFrame(np.isfinite(raw[HOUSE_COLUMNS]),index=raw.index,columns=HOUSE_COLUMNS)
        house_original_masks[name]=mask
        export_table(mask.rename_axis("observation_time"),f"house_{name.lower()}_original_masks.csv",index=True)
        for column in HOUSE_COLUMNS:
            missing=~mask[column]
            groups=missing.ne(missing.shift()).cumsum()
            lengths=missing.loc[missing].groupby(groups[missing]).size()
            house_audit_rows.append({"period":name,"column":column,"rows":len(raw),"missing":int(missing.sum()),
                                    "longest_gap_steps":int(lengths.max()) if len(lengths) else 0})
    house_audit=pd.DataFrame(house_audit_rows)
    display(house_audit);export_table(house_audit,"house_data_audit.csv")
    fig,axes=plt.subplots(3,2,figsize=(12,7),sharex="col")
    for col,(name,raw) in enumerate(house_raw.items()):
        axes[0,col].plot(raw.index,raw[HOUSE_COLUMNS[0]],color="black",label="Indoor")
        axes[0,col].plot(raw.index,raw[HOUSE_COLUMNS[1]],color="0.5",label="Outdoor")
        axes[0,col].set_title(name);axes[0,col].legend(fontsize=9,ncol=2)
        axes[1,col].plot(raw.index,raw[HOUSE_COLUMNS[2]]/1000,color="tab:orange")
        axes[2,col].plot(raw.index,raw[HOUSE_COLUMNS[3]],color="tab:green");time_axis(axes[2,col])
        for ax in axes[:,col]:ax.tick_params(axis="x",labelbottom=False)
        axes[2,col].tick_params(axis="x",labelbottom=True)
        axes[2,col].xaxis.set_major_locator(mdates.DayLocator())
        axes[2,col].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        axes[2,col].set_xlabel("Observation date")
    axes[0,0].set_ylabel("Temperature (°C)");axes[1,0].set_ylabel("Power proxy (kW)");axes[2,0].set_ylabel("Solar (W/m²)")
    fig.tight_layout()
    if EXPORT_ARTEFACTS and (EXTRA_EXPORT_NAMES is None or "extra_house_raw_signals" in EXTRA_EXPORT_NAMES):
        FIGURE_DIR.mkdir(parents=True,exist_ok=True)
        for ext in ["png","svg"]:fig.savefig(FIGURE_DIR/f"extra_house_raw_signals.{ext}",dpi=220,bbox_inches="tight")
    if EXPORT_ARTEFACTS:
        house_source_files=["data/tutorial_6_2/10mins_solpap_2016.csv","data/tutorial_6_2/10mins_solpap_2017.csv"]
        pd.DataFrame([{"slide":27,"png":"figures/week6_day2/extra_house_raw_signals.png",
            "svg":"figures/week6_day2/extra_house_raw_signals.svg",
            "source_files":";".join(house_source_files),
            "source_sha256":";".join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in house_source_files),
            "caption":"Original November and June measurements: indoor/outdoor temperature, electricity-plus-gas power proxy and solar irradiance; missing observations retained."}]).to_csv(ROOT/"teaching/week6_day2_supplemental_placements.csv",index=False)
    plt.show();plt.close(fig)
    ''')
    code('house-transitions','''
    # Supplied transfer demonstration; independent of the synthetic student fit.
    house_tables={}
    HOUSE_FEATURES=["temperature_difference_c","power_proxy_kw","solar_proxy_kw_m2"]
    HOUSE_LOWER=np.array([1e-6,1e-6,0.]);HOUSE_UPPER=np.array([.2,.5,2.])
    for name,raw in house_raw.items():
        table=pd.DataFrame(index=raw.index)
        table["predictor_time"]=raw.index
        table["target_time"]=raw.index.to_series().shift(-1)
        table["indoor_temperature_c"]=raw[HOUSE_COLUMNS[0]]
        table["outdoor_temperature_c"]=raw[HOUSE_COLUMNS[1]]
        table["power_proxy_kw"]=raw[HOUSE_COLUMNS[2]]/1000
        table["solar_proxy_kw_m2"]=raw[HOUSE_COLUMNS[3]]/1000
        table["next_temperature_c"]=table.indoor_temperature_c.shift(-1)
        table["temperature_change_c"]=table.next_temperature_c-table.indoor_temperature_c
        table["temperature_difference_c"]=table.outdoor_temperature_c-table.indoor_temperature_c
        table["original_eligible"]=np.isfinite(table[["indoor_temperature_c","next_temperature_c","outdoor_temperature_c","power_proxy_kw","solar_proxy_kw_m2"]]).all(axis=1)&(table.target_time-table.predictor_time).eq(pd.Timedelta(minutes=10))
        house_tables[name]=table.iloc[:-1].copy()
    winter_transitions=house_tables["November"].loc[house_tables["November"].original_eligible]
    house_fit=lsq_linear(winter_transitions[HOUSE_FEATURES].to_numpy(),winter_transitions.temperature_change_c.to_numpy(),bounds=(HOUSE_LOWER,HOUSE_UPPER))
    if not house_fit.success:raise RuntimeError(house_fit.message)
    house_coefficients=house_fit.x.copy()
    house_model_record=pd.DataFrame([{"fit_period":"November only","N_fit":len(winter_transitions),
        "b":house_coefficients[0],"c_proxy_K_per_kW":house_coefficients[1],"g_proxy":house_coefficients[2],
        "R_proxy_K_per_kW":house_coefficients[1]/house_coefficients[0],
        "C_proxy_kWh_per_K":(1/6)/house_coefficients[1],"tau_hours":(1/6)/house_coefficients[0],
        "active_bounds":str(house_fit.active_mask.tolist())}])
    display(house_model_record);export_table(house_model_record,"house_frozen_winter_model.csv")
    ''')
    code('house-predict','''
    def predict_house_segments(table,coefficients):
        """Propagate through observed inputs; mark every measured segment reset."""
        result=table.copy()
        result["one_step_c"]=result.indoor_temperature_c+result[HOUSE_FEATURES]@coefficients
        result["rollout_c"]=np.nan
        result["segment_id"]=-1
        result["reset_at_predictor"]=False
        predicted=np.nan;segment=-1
        loss,power,solar=coefficients
        for index,row in result.iterrows():
            inputs=row[["outdoor_temperature_c","power_proxy_kw","solar_proxy_kw_m2"]].to_numpy(dtype=float)
            if not np.isfinite(inputs).all():
                predicted=np.nan
                continue
            if not np.isfinite(predicted):
                if not np.isfinite(row.indoor_temperature_c):continue
                predicted=float(row.indoor_temperature_c);segment+=1
                result.loc[index,"reset_at_predictor"]=True
            predicted=predicted+loss*(row.outdoor_temperature_c-predicted)+power*row.power_proxy_kw+solar*row.solar_proxy_kw_m2
            result.loc[index,["rollout_c","segment_id"]]=[predicted,segment]
        result["common_score_mask"]=result.original_eligible&np.isfinite(result.rollout_c)
        result["one_step_residual_c"]=result.next_temperature_c-result.one_step_c
        result["rollout_residual_c"]=result.next_temperature_c-result.rollout_c
        return result

    house_predictions={};house_score_rows=[];house_segment_rows=[]
    for name,table in house_tables.items():
        predicted=predict_house_segments(table,house_coefficients)
        house_predictions[name]=predicted
        use=predicted.loc[predicted.common_score_mask].set_index("target_time")
        for method,column in [("One-step","one_step_c"),("Segmented rollout","rollout_c")]:
            house_score_rows.append({"period":name,"method":method,**temperature_metrics(use.next_temperature_c,use[column])})
        for segment,group in predicted.loc[predicted.segment_id>=0].groupby("segment_id"):
            house_segment_rows.append({"period":name,"segment_id":segment,"start":group.predictor_time.iloc[0],
                "end":group.target_time.iloc[-1],"N_transitions":len(group),"duration_hours":len(group)/6})
        export_table(predicted,f"house_{name.lower()}_predictions.csv")
    house_scores=pd.DataFrame(house_score_rows);house_segments=pd.DataFrame(house_segment_rows)
    display(house_scores);display(house_segments.groupby("period").agg(segments=("segment_id","size"),longest_hours=("duration_hours","max")))
    export_table(house_scores,"house_scores.csv");export_table(house_segments,"house_rollout_segments.csv")
    ''')
    code('house-transfer-figure','''
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,(name,raw) in zip(axes,house_raw.items()):
        predicted=house_predictions[name]
        ax.plot(raw.index,raw[HOUSE_COLUMNS[0]],color="black",label="Measured (original)")
        ax.plot(predicted.target_time,predicted.one_step_c,color="tab:blue",alpha=.7,label="One-step (10 min)")
        for number,(segment,group) in enumerate(predicted.loc[predicted.segment_id>=0].groupby("segment_id")):
            ax.plot(group.target_time,group.rollout_c,color="tab:orange",label="Segmented rollout" if number==0 else None)
        resets=predicted.loc[predicted.reset_at_predictor]
        ax.scatter(resets.predictor_time,resets.indoor_temperature_c,marker="|",s=30,color="tab:orange",label="Explicit reset")
        N=int(predicted.common_score_mask.sum());segments=int(predicted.reset_at_predictor.sum())
        ax.set(title=f"{name}: N={N}; {segments} segments",ylabel="Indoor temperature (°C)",xlabel="Target / state time")
        ax.set_xlim(raw.index[0],raw.index[-1])
        legend_above(ax,2);ax.set_title(ax.get_title(),pad=50);time_axis(ax)
    save_figure(fig,"F62-10",["data/tutorial_6_2/10mins_solpap_2016.csv","data/tutorial_6_2/10mins_solpap_2017.csv"],
        "11–15 November 2016; 10–14 June 2017","°C;kW electricity+gas proxy;kW/m²",
        "Frozen November coefficients; original-observation headline scores; no imputation",
        counts="720 observations/period; common targets "+str(house_scores.groupby("period").N.first().to_dict()))
    ''')
    md('house-interpret','''
    **Interpret:** the measured-state resets make these segmented rollouts shorter
    experiments. Their scores are not five-day fixed-origin forecast scores.
    Optional E1 links the segments by interpolating missing outdoor inputs; this
    also removes resets and changes the prediction information set.
    ''')
    md('i8',r'''
    ## I8. Three model roles and Week 7 handover (slides 29–31)

    1. **Student fit:** effective model estimated on the 21-day fit period and
       checked on later observations and declared horizons.
    2. **Common instructor model:** fixed shared coefficients for downstream
       comparisons. The existing `verified_controller_model.json` was fitted on
       all 28 synthetic days; its last seven days are not held out.
    3. **Reference plant:** richer simulated response used to evaluate actions.

    For electrical input, $A=1-b$, $B=c\,COP$, $E=b$ with an explicit COP
    assumption. Do not replace your fitted coefficients with the common model.
    Preserve its JSON schema and fields. Model adequacy depends on the intended
    horizon, margins and operating conditions, not a fit score alone.

    **Conclusion:** Estimated R and C depend on the data, estimation period and
    model assumptions. Bounds enforce specified limits but do not establish unique
    physical values. Accurate one-step predictions can coexist with substantial
    rollout error. Evaluate later-period performance at the horizon required for
    the intended use.
    ''')
    code('provenance','''
    common_model=json.loads((ROOT/"data/reference_building/verified_controller_model.json").read_text())
    display(pd.Series(common_model,name="Common instructor model (all 28 days)"))
    provenance={"student_model":{"source":"data/reference_building/identification_data.csv",
        "fit_start":str(fit_observations.index[0]),"fit_end":str(fit_observations.index[-1]),"N_fit_transitions":len(fit_transitions),
        "validation_start":str(actual.index[0]),"validation_end":str(actual.index[-1]),"N_common_targets":len(actual),
        "b":float(b),"c_K_per_kW":float(c),"dt_hours":dt,"validation_inputs":"recorded, ex-post conditional"},
        "common_model":{"file":"data/reference_building/verified_controller_model.json","fit_provenance":"all 28 synthetic days; no independent seven-day holdout",
                         "sha256":hashlib.sha256((ROOT/"data/reference_building/verified_controller_model.json").read_bytes()).hexdigest()},
        "real_house":{"source":"https://discovery.ucl.ac.uk/id/eprint/10087216/","teaching_reference":"https://github.com/akylasstrat/BENV0092/blob/main/exercise_3_RC_model.ipynb",
            "fit":"originally observed November transitions only","June":"frozen winter fit evaluated before post-hoc refit",
            "heat_input":"electricity+gas power proxy, not delivered space heat","rollout":"explicit observed-input segments; no target-triggered reset"}}
    if EXPORT_ARTEFACTS:(OUTPUT_DIR/"model_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf8")
    ''')
    code('core-manifest','''
    if EXPORT_ARTEFACTS:
        pd.DataFrame(figure_manifest).sort_values("id").to_csv(ROOT/"teaching/week6_day2_figure_manifest.csv",index=False)
    ''')
    md('optional',r'''
    ## Optional real-data extension E1–E4 (slides 33–36; 35–45 min)

    The supplied transfer example above is complete without these exercises.
    Use this section to investigate missingness, seasonal transfer and effective
    parameter sensitivity. June is evaluated with the frozen winter fit **before**
    its later descriptive refit. No optional estimate replaces the headline model.

    ### E1. Inspect gaps and test limited interpolation

    Retain the original values and masks. The observed outdoor gaps last at most
    three 10-minute intervals. As an explicit sensitivity assumption, interpolate
    only whole internal runs of at most three missing readings within each period.
    Never fill endpoints, partly fill a longer run or overwrite observations.
    This policy is not proof that the missing weather followed a linear path.
    ''')
    task('E1','''
    short_filled={}
    for name,raw in house_raw.items():
        cleaned=raw[HOUSE_COLUMNS].copy()
        for column in HOUSE_COLUMNS:
            missing=cleaned[column].isna()
            groups=missing.ne(missing.shift()).cumsum()
            allowed=pd.Series(False,index=cleaned.index)
            for _,run in cleaned.loc[missing,column].groupby(groups[missing]):
                positions=cleaned.index.get_indexer(run.index)
                if len(run)<=3 and positions[0]>0 and positions[-1]<len(cleaned)-1:
                    allowed.loc[run.index]=True
            candidate=cleaned[column].interpolate(method="time",limit_area="inside")
            cleaned.loc[allowed,column]=candidate.loc[allowed]
        short_filled[name]=cleaned
    ''','''
    # TODO E1: fill only complete internal runs of at most 3 readings, separately per period/column.
    # Preserve house_raw and house_original_masks; create short_filled as a sensitivity copy.
    raise NotImplementedError("Extension E1: limited internal gaps only, never endpoint extrapolation.")
    ''',True)
    code('gap-sensitivity','''
    # Input-imputation sensitivity only: still score originally observed common targets.
    gap_sensitivity_rows=[]
    gap_sensitivity_predictions={}
    for name,cleaned in short_filled.items():
        sensitivity=house_tables[name].copy()
        sensitivity["outdoor_temperature_c"]=cleaned[HOUSE_COLUMNS[1]].reindex(sensitivity.index)
        # Keep original indoor temperatures/masks; only the missing outdoor forcing is filled here.
        sensitivity["temperature_difference_c"]=sensitivity.outdoor_temperature_c-sensitivity.indoor_temperature_c
        predicted=predict_house_segments(sensitivity,house_coefficients)
        gap_sensitivity_predictions[name]=predicted
        use=predicted.loc[house_predictions[name].common_score_mask].set_index("target_time")
        gap_sensitivity_rows.append({"period":name,"method":"Limited outdoor-input imputation; frozen model",
                                    "N_segments":int(predicted.reset_at_predictor.sum()),
                                    **temperature_metrics(use.next_temperature_c,use.rollout_c)})
        export_table(predicted,f"optional_{name.lower()}_input_gap_predictions.csv")
    gap_sensitivity_scores=pd.DataFrame(gap_sensitivity_rows)
    display(gap_sensitivity_scores);export_table(gap_sensitivity_scores,"optional_input_gap_sensitivity.csv")
    fig,axes=plt.subplots(1,2,figsize=(12,5.5))
    for ax,(name,predicted) in zip(axes,gap_sensitivity_predictions.items()):
        raw=house_raw[name]
        ax.plot(raw.index,raw[HOUSE_COLUMNS[0]],color="black",label="Measured (original)")
        ax.plot(predicted.target_time,predicted.rollout_c,color="tab:orange",label="Continuous rollout; weather imputed")
        ax.set(title=name,ylabel="Indoor temperature (°C)",xlabel="Target time",xlim=(raw.index[0],raw.index[-1]))
        legend_above(ax,1);ax.set_title(name,pad=48);time_axis(ax)
    fig.tight_layout()
    if EXPORT_ARTEFACTS and (EXTRA_EXPORT_NAMES is None or "extra_house_input_gap_sensitivity" in EXTRA_EXPORT_NAMES):
        for ext in ["png","svg"]:fig.savefig(FIGURE_DIR/f"extra_house_input_gap_sensitivity.{ext}",dpi=220,bbox_inches="tight")
    plt.show();plt.close(fig)
    ''')
    md('gap-protocol-warning','''
    These scores use the same originally observed targets as the headline example.
    Interpolated temperature targets are never treated as truth. Filling the
    forcing gaps also joins rollout segments and removes measured-state resets;
    the change in error therefore reflects **both** the forcing assumption and
    the initialisation protocol. It does not measure interpolation accuracy alone.
    ''')
    md('e2',r'''
    ### E2. Reproduce the fixed November model and transfer check

    Fit the three-predictor bounded model yourself using the original-observation
    November rows. Compare with the supplied example, then evaluate June unchanged.
    Explain what the proxy and omitted gains prevent you from claiming about R,C.
    ''')
    task('E2','''
    extension_fit=lsq_linear(winter_transitions[HOUSE_FEATURES].to_numpy(),winter_transitions.temperature_change_c.to_numpy(),bounds=(HOUSE_LOWER,HOUSE_UPPER))
    if not extension_fit.success:raise RuntimeError(extension_fit.message)
    extension_coefficients=extension_fit.x
    extension_june=predict_house_segments(house_tables["June"],extension_coefficients)
    use=extension_june.loc[extension_june.common_score_mask].set_index("target_time")
    extension_june_scores=temperature_metrics(use.next_temperature_c,use.rollout_c)
    display(extension_june_scores)
    ''','''
    # TODO E2: fit original November transitions with the fixed bounds, then apply unchanged to June.
    raise NotImplementedError("Extension E2: retain the winter fit for independent seasonal transfer.")
    ''',True)
    md('e3',r'''
    ### E3. Descriptive fit-period sensitivity

    After reporting the frozen-model June scores, compare full November, first
    and final **clock-time** 48 h of November, and a post-hoc full June refit.
    Both predictor and target must belong to each window. Do not take the first
    288 already-filtered rows and call that 48 h. Complete the fit call and
    parameter expressions in the supplied loop. The June row is in-sample,
    post-hoc evidence, not an improved independent validation result.
    ''')
    code('sensitivity-windows','''
    winter_start=house_raw["November"].index[0];winter_end=house_raw["November"].index[-1]
    winter_table=house_tables["November"]
    first48=winter_table.loc[(winter_table.predictor_time>=winter_start)&(winter_table.target_time<winter_start+pd.Timedelta(hours=48))]
    final48=winter_table.loc[(winter_table.predictor_time>=winter_end-pd.Timedelta(hours=48)+pd.Timedelta(minutes=10))&(winter_table.target_time<=winter_end)]
    sensitivity_windows={"Nov all":winter_table,"Nov first 48h":first48,"Nov final 48h":final48,"June post-hoc":house_tables["June"]}
    ''')
    task('E3','''
    sensitivity_rows=[]
    for name,table in sensitivity_windows.items():
        valid=table.loc[table.original_eligible]
        result=lsq_linear(valid[HOUSE_FEATURES].to_numpy(),valid.temperature_change_c.to_numpy(),bounds=(HOUSE_LOWER,HOUSE_UPPER))
        if not result.success:raise RuntimeError(result.message)
        loss,power,solar=result.x
        sensitivity_rows.append({"window":name,"N_fit":len(valid),"start":valid.predictor_time.min(),"end":valid.target_time.max(),
            "b":loss,"c_proxy":power,"g_proxy":solar,"R_proxy_K_per_kW":power/loss,
            "C_proxy_kWh_per_K":(1/6)/power,"tau_hours":(1/6)/loss,"active_bounds":str(result.active_mask.tolist()),
            "role":"post-hoc in-sample" if name=="June post-hoc" else "November fitting sensitivity"})
    sensitivity_table=pd.DataFrame(sensitivity_rows)
    ''','''
    # TODO E3: fit each original-observation window and recover effective R,C,tau; retain counts/bounds.
    raise NotImplementedError("Extension E3: label the June refit post-hoc and retain the original transfer result.")
    ''',True)
    code('sensitivity-figure','''
    display(sensitivity_table);export_table(sensitivity_table,"optional_fit_period_sensitivity.csv")
    fig,axes=plt.subplots(1,3,figsize=(12,5.5))
    labels=["Nov\\nall","Nov\\nfirst\\n48 h","Nov\\nfinal\\n48 h","June refit\\n(post-hoc)"]
    for ax,column,ylabel in zip(axes,["R_proxy_K_per_kW","C_proxy_kWh_per_K","tau_hours"],
                              ["Effective R (K/kW)","Effective C (kWh/K)","Time constant (h)"]):
        ax.bar(range(4),sensitivity_table[column],color=["tab:blue"]*3+["0.5"])
        ax.set(xticks=range(4),xticklabels=labels,ylabel=ylabel)
    save_figure(fig,"F62-11",["data/tutorial_6_2/10mins_solpap_2016.csv","data/tutorial_6_2/10mins_solpap_2017.csv"],
        "Four documented clock-time fitting windows","K/kW;kWh/K;h","Original-observation fits; June post-hoc","Optional E3 fitting",
        counts="N_fit by window: "+str(dict(zip(sensitivity_table.window,sensitivity_table.N_fit))))
    # Rounded display only; all reported values come from these computed fits.
    display(sensitivity_table.loc[sensitivity_table.window.ne("June post-hoc"),
        ["window","N_fit","R_proxy_K_per_kW","C_proxy_kWh_per_K","tau_hours","active_bounds"]].round(3))
    november_comparison=sensitivity_table.set_index("window")
    for column,label in [("R_proxy_K_per_kW","R (K/kW)"),("C_proxy_kWh_per_K","C (kWh/K)"),("tau_hours","tau (h)")]:
        values=november_comparison.loc[["Nov all","Nov first 48h","Nov final 48h"],column]
        print(f"{label}: full {values.iloc[0]:.3f}; first 48 h {values.iloc[1]:.3f}; final 48 h {values.iloc[2]:.3f}.")
    ''')
    md('sensitivity-interpret','''
    **Interpret F62-11:** compare the computed full, first and final November
    estimates of R,C and tau, together with sample counts and active bounds.
    Parameter sensitivity need not be equally large for every parameter. Input
    variation, correlations, unmodelled gains and noise differ across estimation
    periods; finite-data estimates under a heat proxy need not be unique fabric
    properties. Their variation alone does not demonstrate changing building fabric.
    The four fits support no significance or confidence claims. The grey
    **June refit (post-hoc)** follows frozen-model June evaluation and never
    replaces its coefficients or scores. This optional evidence adds no core dependency.
    ''')
    md('e4',r'''
    ### E4. Evidence and possible model changes

    Use the newly computed table; do not repeat earlier rounded estimates.
    Which window changes the effective parameters most? Did any bound activate?
    Why need these differences not imply changing building fabric? Compare
    observed-input segmented rollouts with E1's labelled forcing-imputation
    sensitivity. What can indoor/outdoor sensors, delivered heating and solar
    measurements add? Propose one intercept/gain/state extension and the new
    final evaluation data needed to judge it. No advanced estimator is required.
    ''')
    code('manifest','''
    if EXPORT_ARTEFACTS:
        pd.DataFrame(figure_manifest).sort_values("id").to_csv(ROOT/"teaching/week6_day2_figure_manifest.csv",index=False)
    ''')
    md('references','''
    ### References and source credits (slide 32)

    - Synthetic data assumptions: [reference data README](data/reference_building/README.md).
    - Real data: [Hollick and Wingfield (2018), UCL dataset](https://discovery.ucl.ac.uk/id/eprint/10087216/),
      DOI 10.14324/000.ds.10087216; [local data README](data/tutorial_6_2/README.md).
    - Teaching reference: [BENV0092 Exercise 3](https://github.com/akylasstrat/BENV0092/blob/main/exercise_3_RC_model.ipynb).
      The present workflow preserves original masks and evaluates winter-to-summer transfer before refitting June.
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
    print('Built Tutorial 6.2 pair only; six core tasks and three optional coding exercises.')
