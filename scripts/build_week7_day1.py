"""Scoped authoring for Tutorial 7.1; student-facing explanations and code."""
from pathlib import Path
import csv
import textwrap
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]
STEM='tutorial_7_1_building_optimisation'
TASKS=[('A',11,15),('B',21,8),('C',24,10),('D',28,10),('E',36,12),('F',40,7)]
SLIDE_MINUTES=[2,2,3,3,4,4,4,4,5,4,15,5,4,4,5,5,5,6,4,4,8,5,4,10,4,4,4,10,3,3,3,3,5,4,4,12,4,3,4,7,2,2]
FIGURES=[
    ('F71-01', 15, 'w7d1_F01_planning_case', 'Supplied outdoor temperature, endpoint comfort bounds and ToU prices; known future inputs.'),
    ('F71-02', 12, 'w7d1_F02_toy_feasible_region', 'Dimensionless graphical LP: labelled half-planes, feasible intersection, contours, improvement direction and optimum; show after Task A.'),
    ('F71-03', 17, 'w7d1_F03_thermal_time_index', 'Two interval actions and three boundary states; interval comfort is checked at the endpoint.'),
    ('F71-04', 25, 'w7d1_F04_hard_schedule', 'Hard-comfort ToU plan: 48 interval powers and 49 boundary temperatures.'),
    ('F71-05', 'Notebook only', 'w7d1_F05_hard_soft_comparison', 'Hard versus lower-soft traces at gamma=0.25; bill and penalty reported separately in metrics.'),
    ('F71-06', 37, 'w7d1_F06_penalty_tradeoff', 'Bill versus lower shortfall for the expanded gamma sweep, with the hard-comfort baseline.'),
    ('F71-08', 29, 'w7d1_F08_tariff_cross_evaluation', 'Flat-optimised and ToU-optimised powers and temperatures under identical hard physical constraints.'),
    ('F71-09', 44, 'w7d1_F09_initial_error_effect', 'Optional signed initial-observation error; fixed-plan replay starts from true 20 C; both violation signs and failures retained.'),
    ('F71-11', 30, 'w7d1_F11_tariff_bill_table', 'Two optimisation tariffs by two evaluation tariffs; all four bills in pounds; native table CSV included.'),
    ('F71-12', 40, 'w7d1_F12_peak_objective_comparison', 'Cost/peak schedules with bill, energy, mean endpoint temperature, peak power, maximum interval energy and both comfort components.'),
    ('F71-13', 27, 'w7d1_F13_tariff_inputs', 'Supplied ToU price profile and its positive arithmetic-mean flat comparator.'),
    ('F71-14', 38, 'w7d1_F14_penalty_sensitivity', 'Bill, penalised objective and lower shortfall versus gamma; piecewise changes and plateaus are retained.'),
]
FIGURE_TASKS={'F71-01': 'B', 'F71-02': 'A', 'F71-03': 'B', 'F71-04': 'C', 'F71-05': 'E', 'F71-06': 'E', 'F71-08': 'D', 'F71-09': 'Optional', 'F71-11': 'D', 'F71-12': 'F', 'F71-13': 'D', 'F71-14': 'E'}


def make_notebook(solution=False):
    cells=[]
    def md(key,source):cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(),id='w71-'+key))
    def code(key,source,tags=None):
        metadata={'tags':tags or ['supplied']}
        if key=='result-checks':metadata['jupyter']={'source_hidden':True}
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(),id='w71-'+key,metadata=metadata))
    def task(letter,answer,scaffold):code('task-'+letter.lower(),answer if solution else scaffold,['task',letter])

    md('purpose',r'''
    # Tutorial 7.1: Optimisation foundations for building operation

    Formulate and check an optimisation problem, then compare operating schedules
    under common physical conditions. Begin with a dimensionless graphical example;
    use the supplied building model for the subsequent heating exercises.

    **Outcomes:** distinguish variables, parameters, objectives and constraints;
    formulate a linear program; verify a returned schedule; compare tariffs fairly;
    interpret comfort slack and a peak-power objective.

    | Block | Slides | Minutes | Focus |
    |---|---|---:|---|
    | 1 | 1–11 | 50 | Terminology, graphical LP and Task A |
    | 2 | 12–21 | 50 | Heating brief, hard formulation and Task B |
    | Break | after 21 | 15–20 additional | Save and pause |
    | 3 | 22–31 | 50 | Checked solve and flat/ToU comparison, Tasks C–D |
    | 4 | 32–42 | 50 | Lower-soft comfort and peak minimisation, Tasks E–F |

    References are on slide 43. Slides 44-46 are optional. This notebook accompanies Lecture 7.1. Run from the repository root in the supplied environment.
    ''')
    code('setup',r'''
    from pathlib import Path
    import hashlib
    import json
    import warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import cvxpy as cp
    from matplotlib.patches import Polygon, FancyBboxPatch
    from IPython.display import display

    ROOT=Path.cwd()
    DATA_FILE=ROOT/"data/reference_building/operation_conditions.csv"
    MODEL_FILE=ROOT/"data/reference_building/verified_controller_model.json"
    if not DATA_FILE.is_file() or not MODEL_FILE.is_file():
        raise FileNotFoundError("Run from the repository root containing data/reference_building.")
    available_solvers=cp.installed_solvers()
    SOLVER=next((s for s in ["CLARABEL","SCIPY","OSQP","SCS"] if s in available_solvers),None)
    if SOLVER is None:raise RuntimeError("An installed open-source CVXPY solver is required.")
    SOLVER_OPTIONS={"tol_gap_abs":1e-9,"tol_gap_rel":1e-9,"tol_feas":1e-9} if SOLVER=="CLARABEL" else {}
    FEASIBILITY_TOL=1e-6  # numerical residual/bound check, in the stated variable units
    COMFORT_COUNT_TOL=1e-6  # degrees C; counts only, never round magnitudes
    TRUE_T0=20.0
    EXPORT_ARTEFACTS=True
    FIGURE_DIR=ROOT/"figures/week7_day1"
    OUTPUT_DIR=ROOT/"teaching/week7_day1_outputs"
    plt.rcParams.update({"figure.dpi":110,"font.size":12,"axes.grid":True,
        "grid.alpha":.2,"svg.fonttype":"none","savefig.facecolor":"white"})
    COLOURS={"hard":"tab:blue","soft":"tab:orange","flat":"tab:green","tou":"tab:blue","peak":"tab:purple","truth":"black"}
    print("CVXPY",cp.__version__,"installed solvers:",available_solvers,"selected:",SOLVER)
    ''')
    md('helpers-text',r'''
    ### Supplied plotting/export helpers

    Set `EXPORT_ARTEFACTS=False` to avoid writing files. Plot states at boundaries
    and powers over intervals. The supplied helpers are used as each result becomes
    available; they do not load precomputed answers.
    ''')
    code('helpers',"FIGURE_SPECS="+repr(FIGURES)+"\nFIGURE_TASKS="+repr(FIGURE_TASKS)+"\n"+textwrap.dedent(r'''
    figure_manifest=[]
    source_paths=["data/reference_building/operation_conditions.csv","data/reference_building/verified_controller_model.json"]
    def export_table(frame,name,index=False):
        if EXPORT_ARTEFACTS:
            OUTPUT_DIR.mkdir(parents=True,exist_ok=True);frame.to_csv(OUTPUT_DIR/name,index=index)

    def save_figure(fig,figure_id,case="Synthetic one-day planning case",counts=None):
        spec=next(row for row in FIGURE_SPECS if row[0]==figure_id)
        diagram=figure_id in {"F71-02","F71-03"}
        sources=["scripts/build_week7_day1.py"] if diagram else source_paths
        fig.tight_layout()
        if EXPORT_ARTEFACTS:
            FIGURE_DIR.mkdir(parents=True,exist_ok=True)
            for ext in ["png","svg"]:fig.savefig(FIGURE_DIR/f"{spec[2]}.{ext}",dpi=220)
            usage="Notebook only" if not isinstance(spec[1],int) else ("Optional slide" if spec[1]>42 else "Core slide")
            figure_manifest.append({"id":figure_id,"slide":spec[1],"use":usage,
                "task":FIGURE_TASKS[figure_id],"png":f"figures/week7_day1/{spec[2]}.png",
                "svg":f"figures/week7_day1/{spec[2]}.svg","notebook":"tutorial_7_1_building_optimisation_solution.ipynb",
                "placement":"12.5 x 3.5 inch frame; preserve aspect ratio","source_files":";".join(sources),
                "source_sha256":";".join(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources),
                "case":case,"period":"Illustrative" if diagram else f"{interval_start[0]} to {interval_end[-1]}",
                "dt_hours":"not applicable" if figure_id in {"F71-02","F71-10"} else dt,
                "N_intervals":0 if figure_id in {"F71-02","F71-10"} else (2 if figure_id=="F71-03" else N),
                "counts":counts or ("diagram" if diagram else "48 inputs; 49 boundary states"),
                "units":"dimensionless" if figure_id=="F71-02" else ("model roles" if figure_id=="F71-10" else "kW;kWh;h;degrees C;GBP;GBP/kWh"),
                "assumptions":"Known future inputs; fixed COP; endpoint comfort; final comfort only" if not diagram else "Illustrative diagram, not measured data",
                "caption":spec[3]})
        plt.show();plt.close(fig)

    def clock_axis(ax):
        ax.set_xlim(state_time[0],state_time[-1])
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0,24,4)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    def comfort_lines(ax):
        ax.plot(state_time[1:],lower,".--",color="black",markersize=3,label="Endpoint comfort bounds")
        ax.plot(state_time[1:],upper,".--",color="black",markersize=3)

    def table_figure(frame,figsize=(12.5,3.5),fontsize=12):
        shown=frame.copy()
        for column in shown.select_dtypes(include="floating"):
            shown[column]=shown[column].map(lambda x:f"{x:.4g}")
        fig,ax=plt.subplots(figsize=figsize);ax.axis("off")
        tab=ax.table(cellText=shown.values,colLabels=shown.columns,loc="center",cellLoc="center")
        tab.auto_set_font_size(False);tab.set_fontsize(fontsize);tab.scale(1,2.4)
        for (row,col),cell in tab.get_celld().items():
            cell.set_edgecolor("0.8")
            if row==0:cell.set_facecolor("0.92")
        return fig

    def export_schedule(result,name):
        if not result["success"]:return
        frame=intervals.copy()
        frame["power_electric_kw"]=result["power_kw"]
        frame["lower_slack_c"]=result["slack_c"]
        frame["predicted_endpoint_c"]=result["temperature_c"][1:]
        frame["planning_price_gbp_per_kwh"]=result["prices"]
        export_table(frame,name+"_intervals.csv")
        export_table(pd.DataFrame({"state_time":state_time,"predicted_temperature_c":result["temperature_c"]}),name+"_states.csv")
    ''').strip())
    md('toy-text',r'''
    ## O1. Task A — inequalities, contours and the optimum (slide 11; 15 min)

    Choose the scalar variables $x,y$ in the following linear program:

    $$
    \begin{aligned}
    \min_{x,y}\quad & x-2y \\
    \text{subject to}\quad & x+y\le 4, \\
    & y\le 3, \\
    & 2x+y\le 6, \\
    & x\ge 0,\quad y\ge 0.
    \end{aligned}
    $$

    The first and third constraints cap different weighted sums; the second caps
    $y$. Nonnegative axes rule out negative decisions. Their intersection is the
    feasible set. A contour $x-2y=c$ joins points with the same objective value;
    smaller values are preferred. This illustrative LP is not a heating bill.

    Complete `feasible` and `objective_grid` on the supplied grid, then identify
    `best_xy` and `best_objective` among feasible points. Use the supplied plot to
    explain which constraints are active (hold at equality) at your proposed optimum.
    Reason graphically **before** running the subsequent solver verification.
    A grid is an approximation; a solver check helps distinguish discretisation
    from the continuous optimum. Not every LP has a unique vertex optimum.
    ''')
    code('graph-grid',r'''
    x_grid=np.linspace(0.,4.5,451)
    y_grid=np.linspace(0.,4.5,451)
    X,Y=np.meshgrid(x_grid,y_grid)
    ''')
    task('A',r'''
    feasible=(X+Y<=4)&(Y<=3)&(2*X+Y<=6)&(X>=0)&(Y>=0)
    objective_grid=X-2*Y
    best_index=np.unravel_index(np.argmin(np.where(feasible,objective_grid,np.inf)),X.shape)
    best_xy=np.array([X[best_index],Y[best_index]])
    best_objective=float(objective_grid[best_index])
    ''',r'''
    # TODO A: combine all five inequalities in feasible; evaluate objective_grid.
    # Use feasible points to obtain best_xy (two coordinates) and best_objective.
    raise NotImplementedError("Task A: shade the feasible intersection, inspect contours and locate the optimum.")
    ''')
    code('toy-figure',r'''
    fig,(ax,key)=plt.subplots(1,2,figsize=(12.5,3.5),gridspec_kw={"width_ratios":[1,1.1]})
    ax.contourf(X,Y,feasible.astype(int),levels=[.5,1.5],colors=["tab:blue"],alpha=.2)
    ax.plot(x_grid,4-x_grid,label="x + y = 4",color="tab:blue")
    ax.axhline(3,label="y = 3",color="tab:orange")
    ax.plot(x_grid,6-2*x_grid,label="2x + y = 6",color="tab:green")
    ax.axhline(0,color="black");ax.axvline(0,color="black")
    contours=ax.contour(X,Y,objective_grid,levels=[-6,-4,-2,0,2],colors="0.5",linestyles="--")
    ax.clabel(contours,fmt="f=%g",fontsize=10)
    ax.scatter(*best_xy,marker="*",s=130,color="tab:red",zorder=5,
        label=f"Grid optimum ({best_xy[0]:g}, {best_xy[1]:g}); f={best_objective:g}")
    ax.annotate("",xy=(3.,4.),xytext=(3.5,3.),
        arrowprops={"arrowstyle":"->","color":"black"})
    ax.set(xlim=(-.15,4.5),ylim=(-.15,4.5),aspect="equal",xlabel="x",
        ylabel="y",title="Feasible set and contours")
    key.axis("off")
    handles,labels=ax.get_legend_handles_labels()
    key.legend(handles,labels,loc="upper left",fontsize=12)
    key.text(.03,.25,"Objective: minimise x − 2y\nImprovement direction: (−1, 2)\nShading: all five inequalities hold",
        transform=key.transAxes,fontsize=12,va="top",linespacing=1.6)
    save_figure(fig,"F71-02",case="Dimensionless graphical LP",counts="2 variables; 5 inequalities; 451 x 451 grid")
    export_table(pd.DataFrame([{"x":best_xy[0],"y":best_xy[1],"objective":best_objective}]),"graphical_grid_result.csv")
    ''')
    code('graph-solver',r'''
    # 1. Declare the two decision variables.
    graph_x=cp.Variable(name="x")
    graph_y=cp.Variable(name="y")

    # 2. Construct the list of constraints.
    graph_constraints=[
        graph_x+graph_y<=4,
        graph_y<=3,
        2*graph_x+graph_y<=6,
        graph_x>=0,
        graph_y>=0,
    ]

    # 3. Define the objective and construct the problem.
    graph_objective=cp.Minimize(graph_x-2*graph_y)
    graph_problem=cp.Problem(graph_objective,graph_constraints)

    # 4. Solve the problem and check that a solution was returned.
    graph_problem.solve(solver=SOLVER,**SOLVER_OPTIONS)
    if graph_problem.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Graphical example: {graph_problem.status}")
    graph_solution=np.array([graph_x.value,graph_y.value],dtype=float)
    if not np.isfinite(graph_solution).all():
        raise RuntimeError("The solver did not return finite values.")

    # 5. Read the optimal variables and objective.
    graph_check=pd.DataFrame({"x":[graph_solution[0]],"y":[graph_solution[1]],
        "objective":[graph_problem.value],"status":[graph_problem.status]})
    display(graph_check)
    export_table(graph_check,"graphical_solver_result.csv")
    ''')
    md('inputs-text',r'''
    ## O2. The common heating case (slides 15–20)

    **Written brief:** Choose a heat-pump power schedule for the next 24 hours that
    minimises electricity cost, respects power capacity and maintains the specified
    indoor-temperature band, starting from the supplied initial state.

    | Brief phrase | Mathematical role |
    |---|---|
    | Choose heat-pump power | $N$ electrical-power variables $P_k$ |
    | Predict indoor temperature | $N+1$ state variables $T_k$, constrained by dynamics |
    | Minimise electricity cost | Objective $\Delta t\sum_k p_kP_k$ |
    | Respect capacity and comfort | Power bounds and endpoint temperature bounds |
    | Start from the supplied state | Equality $T_0=T_{init}$ |
    | Next 24 hours | 48 half-hour intervals and their supplied input profiles |

    Use the first 48 half-hour intervals of the supplied synthetic operating data.
    Future weather, prices and comfort requirements are **assumed known** for this
    deterministic planning exercise. The cost boundary is heat-pump electricity only:
    no PV, battery, base load or standing charge.

    The common instructor JSON supplies $b,c,dt,COP$ and capacity. It was fitted on
    all 28 identification days, separately from the Week 6 student holdout exercise.
    We do not estimate another model here. Units are kW, h, kWh, °C and £; useful heat
    is $COP\,P_k$. Additional gains are omitted from this compact teaching model.

    Interval $k$ runs from $t_k$ to $t_{k+1}$. Its comfort limits $L_k,U_k$ are checked
    at **$T_{k+1}$**. Endpoint checks do not certify within-interval comfort. The initial
    temperature $T_0=20$°C is fixed. The final boundary has comfort limits, without an
    additional terminal-temperature equality.
    ''')
    code('inputs',r'''
    conditions=pd.read_csv(DATA_FILE,index_col="timestamp",parse_dates=True).iloc[:48].copy()
    model=json.loads(MODEL_FILE.read_text(encoding="utf8"))
    dt=float(model["sampling_interval_hours"])
    b=float(model["loss_coefficient_per_step"])
    c=float(model["heat_coefficient_c_per_kw_step"])
    COP=float(model["nominal_cop"])
    P_MAX=float(model["heat_pump_max_electric_kw"])
    N=len(conditions)
    required=["outdoor_temperature_c","comfort_min_c","comfort_max_c","import_price_gbp_per_kwh"]
    expected=pd.date_range(conditions.index[0],periods=48,freq="30min")
    if N!=48 or dt!=.5 or not conditions.index.equals(expected):
        raise ValueError("Inspect the source interval count, timestamp ordering and half-hour model units.")
    if not np.isfinite(conditions[required]).all().all():raise ValueError("Required inputs must be finite.")
    outdoor=conditions.outdoor_temperature_c.to_numpy(copy=True)
    lower=conditions.comfort_min_c.to_numpy(copy=True)
    upper=conditions.comfort_max_c.to_numpy(copy=True)
    dynamic_prices=conditions.import_price_gbp_per_kwh.to_numpy(copy=True)
    if np.any(lower>upper):raise ValueError("Lower comfort exceeds upper comfort.")
    interval_start=pd.DatetimeIndex(conditions.index,name="interval_start")
    interval_end=pd.DatetimeIndex(interval_start+pd.Timedelta(hours=dt),name="interval_end")
    state_time=pd.DatetimeIndex([interval_start[0],*interval_end],name="state_time")
    intervals=pd.DataFrame({"interval_start":interval_start,"interval_end":interval_end,
        "outdoor_c":outdoor,"lower_c":lower,"upper_c":upper,"price_gbp_per_kwh":dynamic_prices})
    display(pd.Series({"start":interval_start[0],"end":interval_end[-1],"intervals":N,
        "boundaries":len(state_time),"dt_h":dt,"true_T0_c":TRUE_T0,"b":b,"c_K_per_kW":c,
        "COP":COP,"capacity_electric_kw":P_MAX},name="Common instructor planning case"))
    ''')
    code('planning-figure',r'''
    export_table(intervals,"planning_inputs.csv")
    fig,axes=plt.subplots(2,1,figsize=(12.5,3.5),sharex=True)
    axes[0].stairs(outdoor,state_time,baseline=None,label="Outdoor (interval input)",color="0.5")
    comfort_lines(axes[0]);axes[0].set(ylabel="Temp. (°C)",title=f" {interval_start[0]:%d %B %Y}")
    axes[0].legend(ncol=2,fontsize=10)
    axes[1].stairs(dynamic_prices,state_time,color="tab:purple")
    axes[1].set(ylabel="Price (£/kWh)",xlabel="Interval time; future inputs assumed known")
    clock_axis(axes[1]);save_figure(fig,"F71-01")
    ''')
    md('formulation',r'''
    ### The hard-comfort linear program

    Let $\mathcal{K}=\{0,\ldots,N-1\}$ index the control intervals. The
    cost-minimising schedule solves

    $$
    \begin{aligned}
    \min_{P,T}\quad & J_{\mathrm{bill}}(P)=\Delta t\sum_{k=0}^{N-1}p_kP_k \\
    \text{subject to}\quad & T_0=T_{\mathrm{init}}, \\
    & T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k,
        && \forall k\in\mathcal{K}, \\
    & 0\le P_k\le P_{\max}, && \forall k\in\mathcal{K}, \\
    & L_k\le T_{k+1}\le U_k, && \forall k\in\mathcal{K}.
    \end{aligned}
    $$

    There are $N=48$ powers and $N+1=49$ boundary states. Predicted temperatures are
    variables, but the dynamics determine their relation to the power schedule.
    The stored discrete coefficient $c$ already incorporates the half-hour step;
    do **not** multiply the recurrence by $dt$ again. The bill does require $dt$
    to convert electrical kW into kWh. With supplied coefficients and inputs this
    is a linear program. The first objective contains only the electricity bill.
    ''')
    code('timeline',r'''
    fig,ax=plt.subplots(figsize=(12.5,3.5));ax.set(xlim=(-.4,2.5),ylim=(-1,1.4));ax.axis("off")
    for k in range(3):
        ax.plot(k,0,"o",color="black",ms=9);ax.text(k,-.3,f"t{k}: T{k}",ha="center",fontsize=14)
    for k in range(2):
        ax.annotate("",xy=(k+1,0),xytext=(k,0),arrowprops={"arrowstyle":"->","color":"tab:blue"})
        ax.text(k+.5,.35,f"P{k}; dt = 0.5 h",ha="center",fontsize=14)
        ax.text(k+1,.9,f"Check L{k}, U{k} at T{k+1}",ha="center",fontsize=12)
    ax.text(0,-.8,"T0 fixed separately",ha="center",fontsize=12)
    ax.text(2,-.8,"Final boundary T_N",ha="center",fontsize=12)
    save_figure(fig,"F71-03",case="Illustrative thermal indexing",counts="2 interval actions; 3 boundary states")
    ''')
    code('parameters',r'''
    # Profiles may vary over the day but are fixed within each solve.
    initial_temperature=cp.Parameter(value=TRUE_T0,name="initial_temperature_c")
    price=cp.Parameter(N,value=dynamic_prices.copy(),name="price_gbp_per_kwh")
    power=cp.Variable(N,nonneg=True,name="electric_power_kw")
    temperature=cp.Variable(N+1,name="boundary_temperature_c")
    ''')
    md('task-b-text',r'''
    ### Task B — translate the heating brief (slide 21; 8 min)

    Use the supplied variables and Parameters. Complete the capacity, dynamics,
    endpoint comfort bounds and bill in the small skeleton below. `base_constraints`
    contains only initial state, capacity and dynamics; `hard_constraints` adds both
    comfort bounds. Formulate the problem here; Task C performs the solve.
    Explain why temperatures have one more element than powers.
    ''')
    task('B',r'''
    base_constraints=[temperature[0]==initial_temperature,power<=P_MAX,
        temperature[1:]==(1-b)*temperature[:-1]+b*outdoor+c*COP*power]
    hard_constraints=base_constraints+[temperature[1:]>=lower,temperature[1:]<=upper]
    bill_expression=dt*cp.sum(cp.multiply(price,power))
    objective=cp.Minimize(bill_expression)
    hard_problem=cp.Problem(objective,hard_constraints)
    ''',r'''
    base_constraints=[temperature[0]==initial_temperature]
    # TODO B: append capacity and the vector dynamics for temperature[1:].
    # hard_constraints = base_constraints + [lower endpoint bound, upper endpoint bound]
    # bill_expression = ...; objective = cp.Minimize(bill_expression)
    # hard_problem = cp.Problem(objective, hard_constraints)
    raise NotImplementedError("Task B: complete hard capacity, dynamics, endpoint comfort and electricity cost.")
    ''')
    md('break',r'''
    Save and take the **15–20-minute break after slide 21**. This break is additional
    to the 200 teaching minutes. The formulated problem retains fixed initial state,
    capacity and both hard endpoint comfort limits.
    ''')
    md('task-c-text',r'''
    ## O3. Task C — solve and verify the hard LP (slide 24; 10 min)

    The formulation defines the problem; CVXPY expresses it in Python; the installed
    numerical solver computes a candidate solution. Complete the solve and an
    independent bill/energy calculation below. Reject failed status and non-finite
    values before reading results. Supplied checks then inspect residuals and copy
    all arrays before any re-solve. A solver status alone is insufficient evidence.
    ''')
    task('C',r'''
    if not hard_problem.is_dcp():raise ValueError("Inspect the formulation's convexity rules.")
    for variable in [power,temperature]:variable.value=None
    hard_problem.solve(solver=SOLVER,**SOLVER_OPTIONS)
    if hard_problem.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Nominal hard case failed: {hard_problem.status}; diagnose before changing comfort.")
    if any(v.value is None or not np.isfinite(v.value).all() for v in [power,temperature]):
        raise RuntimeError("The solver returned missing/non-finite arrays.")
    hard_bill_check=float(sum(float(p)*float(q) for p,q in zip(price.value,power.value))*dt)
    hard_energy_check=float(np.asarray(power.value).sum()*dt)
    ''',r'''
    if not hard_problem.is_dcp():raise ValueError("Inspect the formulation's convexity rules.")
    for variable in [power,temperature]:variable.value=None
    # TODO C: solve with SOLVER and SOLVER_OPTIONS.
    # Reject failed status and missing/non-finite power or temperature arrays.
    # Compute hard_bill_check and hard_energy_check independently from returned power.
    raise NotImplementedError("Task C: solve the hard LP, check status and independently integrate bill and energy.")
    ''')
    md('basic-results-text',r'''
    ### Read the schedule and calculate its main results

    The inputs and model are fixed. After solving, check the written constraints
    and calculate the results directly from the returned arrays. These checks can
    catch coding or indexing mistakes; they are not an uncertainty experiment.

    The calculations below are ordinary NumPy operations. `power.value` contains
    electrical kW; multiplying by `dt` gives kWh per interval. Multiply that energy
    by the corresponding price and sum to obtain the electricity bill.

    Mean indoor temperature below is the average of the **48 predicted interval-end
    temperatures**, excluding the fixed initial state. Peak power is in **kW**;
    maximum energy in one half-hour interval is also shown, in **kWh**.
    ''')
    code('basic-results',r'''
    # Copy the arrays now: a later solve will overwrite CVXPY's .value fields.
    hard_power=np.asarray(power.value).copy()
    hard_temperature=np.asarray(temperature.value).copy()
    hard_prices=np.asarray(price.value).copy()

    # Electricity and cost.
    interval_energy_kwh=hard_power*dt
    total_energy_kwh=interval_energy_kwh.sum()
    electricity_bill_gbp=np.sum(hard_prices*interval_energy_kwh)

    # Temperature and peak demand.
    mean_indoor_temperature_c=hard_temperature[1:].mean()
    peak_power_kw=hard_power.max()
    peak_interval_energy_kwh=interval_energy_kwh.max()

    # Basic checks on the hard-comfort result.
    assert np.all(hard_power>=-FEASIBILITY_TOL)
    assert np.all(hard_power<=P_MAX+FEASIBILITY_TOL)
    assert np.all(hard_temperature[1:]>=lower-FEASIBILITY_TOL)
    assert np.all(hard_temperature[1:]<=upper+FEASIBILITY_TOL)
    expected_next=(1-b)*hard_temperature[:-1]+b*outdoor+c*COP*hard_power
    assert np.allclose(hard_temperature[1:],expected_next,rtol=0,atol=FEASIBILITY_TOL)
    assert abs(hard_temperature[0]-TRUE_T0)<=FEASIBILITY_TOL
    assert abs(hard_problem.value-electricity_bill_gbp)<=FEASIBILITY_TOL

    # Assemble the summary directly from those quantities.
    shortfall=np.maximum(lower-hard_temperature[1:],0.)
    overshoot=np.maximum(hard_temperature[1:]-upper,0.)
    hard_summary=pd.DataFrame([{
        "bill_gbp":electricity_bill_gbp,"energy_kwh":total_energy_kwh,
        "mean_temperature_c":mean_indoor_temperature_c,"peak_electric_kw":peak_power_kw,
        "peak_interval_energy_kwh":peak_interval_energy_kwh,
        "hours_below_band":np.sum(shortfall>COMFORT_COUNT_TOL)*dt,
        "hours_above_band":np.sum(overshoot>COMFORT_COUNT_TOL)*dt,
        "below_band_degree_hours":shortfall.sum()*dt,
        "above_band_degree_hours":overshoot.sum()*dt}])
    SUMMARY_COLUMNS=list(hard_summary.columns)
    SUMMARY_LABELS=["Bill (£)","Energy (kWh)","Mean endpoint T (°C)","Peak power (kW)",
        "Max interval energy (kWh)","Hours below band","Hours above band",
        "Below band (°C h)","Above band (°C h)"]
    display(hard_summary.set_axis(SUMMARY_LABELS,axis=1).round(3))
    export_table(hard_summary,"hard_summary.csv")
    ''')
    md('checks-text',r'''
    ### Small supplied helpers for the following comparisons

    The next cells repeat the same solve, copy and summary operations for several
    cases. A short helper keeps failed solves out of the comparisons. CVXPY's
    `constraint.violation()` reports the numerical distance from satisfying each
    written constraint; we only use it as a supplied tolerance check.

    For comfort reporting, **shortfall** means the number of degrees below the lower
    bound; **overshoot** means degrees above the upper bound. A temperature within
    the band has neither. Degree-hours sum those positive differences times `dt`.
    Durations count only differences larger than `COMFORT_COUNT_TOL=1e-6` °C.
    These definitions concern the 48 endpoints, not the initial state.
    ''')
    code('result-checks',r'''
    def snapshot_result(problem,hard=True,objective_kind="bill",slack_variable=None,penalty_rate=0.,peak_variable=None):
        if problem.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:
            return {"success":False,"status":str(problem.status),"reason":"No solved schedule"}
        variables=[power,temperature]+([] if slack_variable is None else [slack_variable])
        if peak_variable is not None:variables.append(peak_variable)
        if problem.value is None or not np.isfinite(problem.value) or any(v.value is None or not np.isfinite(v.value).all() for v in variables):
            return {"success":False,"status":str(problem.status),"reason":"Missing or non-finite values"}
        p=np.asarray(power.value).copy()
        t=np.asarray(temperature.value).copy()
        s=np.zeros(N) if slack_variable is None else np.asarray(slack_variable.value).copy()
        bill=float(np.sum(price.value*p*dt))
        penalty=float(penalty_rate*np.sum(s)*dt)
        objective_value=float(p.max()) if objective_kind=="peak" else bill+penalty
        constraints_ok=all(np.max(np.abs(constraint.violation()))<=FEASIBILITY_TOL for constraint in problem.constraints)
        objective_ok=abs(float(problem.value)-objective_value)<=FEASIBILITY_TOL
        if problem.status==cp.OPTIMAL_INACCURATE:
            warnings.warn("Solver reports an inaccurate optimum; check the numerical tolerances.",RuntimeWarning)
        result={"success":bool(constraints_ok and objective_ok),"status":str(problem.status),
            "power_kw":p,"temperature_c":t,"slack_c":s,"prices":np.asarray(price.value).copy(),
            "initial_c":float(initial_temperature.value),"gamma":float(penalty_rate),"hard":bool(hard),
            "bill_gbp":bill,"weighted_slack_gbp":penalty,"objective_kind":objective_kind,
            "objective_name":{"bill":"Electricity bill","soft":"Bill plus comfort penalty","peak":"Peak power"}[objective_kind],
            "objective_units":"kW" if objective_kind=="peak" else "GBP",
            "solver_objective":float(problem.value),"recomputed_objective":objective_value}
        if objective_kind!="peak":result["total_objective_gbp"]=objective_value
        if not result["success"]:result["reason"]="The returned schedule failed a numerical check"
        return result

    def solve_case(problem,**options):
        for variable in [power,temperature,*problem.variables()]:variable.value=None
        try:problem.solve(solver=SOLVER,**SOLVER_OPTIONS)
        except cp.error.SolverError as error:
            return {"success":False,"status":"solver_error","reason":str(error)}
        return snapshot_result(problem,**options)

    def require_success(result,name):
        if not result["success"]:raise RuntimeError(f"{name}: {result['status']}; {result.get('reason','')}")

    def deviation_metrics(shortfall,overshoot):
        return {"hours_below_band":float(np.sum(shortfall>COMFORT_COUNT_TOL)*dt),
            "hours_above_band":float(np.sum(overshoot>COMFORT_COUNT_TOL)*dt),
            "below_band_degree_hours":float(shortfall.sum()*dt),
            "above_band_degree_hours":float(overshoot.sum()*dt),
            "total_degree_hours":float((shortfall+overshoot).sum()*dt),
            "max_outside_band_c":float(np.max(shortfall+overshoot))}

    def schedule_metrics(result):
        require_success(result,"Schedule summary")
        p=result["power_kw"];t=result["temperature_c"][1:]
        shortfall=np.maximum(lower-t,0.)
        overshoot=np.maximum(t-upper,0.)
        return {"status":result["status"],"energy_kwh":float(np.sum(p*dt)),
            "bill_gbp":float(np.sum(result["prices"]*p*dt)),
            "mean_temperature_c":float(t.mean()),"peak_electric_kw":float(p.max()),
            "peak_interval_energy_kwh":float((p*dt).max()),
            "weighted_slack_gbp":result["weighted_slack_gbp"],
            "slack_degree_hours":float(result["slack_c"].sum()*dt),
            "objective_name":result["objective_name"],"objective_units":result["objective_units"],
            "solver_objective":result["solver_objective"],"recomputed_objective":result["recomputed_objective"],
            **({"total_objective_gbp":result["total_objective_gbp"]} if "total_objective_gbp" in result else {}),
            **deviation_metrics(shortfall,overshoot)}

    hard_solution=snapshot_result(hard_problem)
    require_success(hard_solution,"Hard ToU case")
    hard_metrics=schedule_metrics(hard_solution)
    export_table(pd.DataFrame([hard_metrics]),"hard_metrics.csv")
    export_schedule(hard_solution,"hard_tou")
    ''')
    code('hard-figure',r'''
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    axes[0].stairs(hard_solution["power_kw"],state_time,color=COLOURS["hard"])
    axes[0].set(ylabel="Heat-pump electrical power (kW)",xlabel="Interval time",title="Cost-optimal schedule: ToU tariff")
    axes[1].plot(state_time,hard_solution["temperature_c"],color=COLOURS["hard"],label="Predicted hard comfort")
    axes[1].scatter([state_time[0]],[TRUE_T0],color="black",marker="s",label="Fixed initial state")
    comfort_lines(axes[1]);axes[1].set(ylabel="Temperature (°C)",xlabel="Boundary time")
    axes[1].legend(fontsize=10,ncol=1)
    for ax in axes:clock_axis(ax)
    save_figure(fig,"F71-04")
    ''')
    md('hard-interpretation',r'''
    ### Interpret the checked plan (slides 25–26)

    Explain one period with heating and one without, using the actual power,
    temperature, comfort limits and tariff. Report energy, bill and residuals
    separately. Do not change bounds or inputs to make a failed nominal case pass.

    The next experiment reuses the same `hard_problem` and its price Parameter.
    Updating a Parameter changes the supplied vector while keeping the mathematical
    structure. Copy each result immediately: a re-solve overwrites CVXPY values.
    Repeated planning comparisons do not implement feedback control.
    ''')
    md('task-d-text',r'''
    ## O4. Task D — flat versus ToU tariffs (slide 28; 10 min)

    Time-of-use (ToU) prices vary across intervals. The controlled flat comparator
    is their arithmetic mean because all intervals have equal duration; this does
    not describe commercial tariff design. Keep model, weather, initial state,
    capacity, comfort, horizon and terminal treatment identical.

    Update only `price.value` and solve the same hard problem for the flat tariff.
    Keep the copied ToU result from Task C. Complete the 2×2 `cross_costs` table:
    rows identify the optimisation tariff; columns identify the evaluation tariff.
    Evaluate each fixed schedule with both profiles. Comparing diagonal bills alone
    changes both schedule and tariff and cannot isolate the value of scheduling.
    ''')
    code('tariff-inputs',r'''
    tou_prices=dynamic_prices.copy()  # original source column retained in conditions
    flat_prices=np.full(N,tou_prices.mean())
    tariffs={"Flat":flat_prices,"ToU":tou_prices}
    initial_temperature.value=TRUE_T0
    fig,ax=plt.subplots(figsize=(12.5,3.5))
    for name,colour in [("Flat",COLOURS["flat"]),("ToU",COLOURS["tou"])]:
        ax.stairs(tariffs[name],state_time,label=name,color=colour)
    ax.set(ylabel="Electricity price (£/kWh)",xlabel="Interval time",title="Mean-matched tariff inputs")
    ax.legend();clock_axis(ax);save_figure(fig,"F71-13",counts="2 price profiles; 48 intervals each")
    export_table(pd.DataFrame({"interval_start":interval_start,"interval_end":interval_end,
        "flat_gbp_per_kwh":flat_prices,"tou_gbp_per_kwh":tou_prices}),"tariff_inputs.csv")
    ''')
    md('parameter-methods',r'''
    ### Two ways to change the input prices

    **1. Build a new problem.** Use the new price array when writing a fresh objective
    and problem, as in the supplied flat-price example below. New power/state
    variables make it clear that this is a separate formulation of the same case.

    **2. Update a Parameter.** Our original `price` is a CVXPY `Parameter`.
    Assign `price.value=flat_prices`, then solve `hard_problem` again. The variables
    and constraints stay the same. Task D uses this shorter approach.

    Changing a NumPy price array does not rewrite an objective already constructed
    from its old values. In either approach, save copies of each result before the
    next solve. Both approaches should give the same optimal bill within tolerance,
    although an LP can have several schedules with that bill.
    ''')
    code('rebuild-example',r'''
    # Method 1: construct a new flat-price problem explicitly.
    rebuilt_power=cp.Variable(N,nonneg=True)
    rebuilt_temperature=cp.Variable(N+1)
    rebuilt_constraints=[
        rebuilt_temperature[0]==TRUE_T0,
        rebuilt_power<=P_MAX,
        rebuilt_temperature[1:]==(1-b)*rebuilt_temperature[:-1]+b*outdoor+c*COP*rebuilt_power,
        rebuilt_temperature[1:]>=lower,
        rebuilt_temperature[1:]<=upper,
    ]
    rebuilt_objective=cp.Minimize(dt*cp.sum(cp.multiply(flat_prices,rebuilt_power)))
    rebuilt_problem=cp.Problem(rebuilt_objective,rebuilt_constraints)
    rebuilt_problem.solve(solver=SOLVER,**SOLVER_OPTIONS)
    if rebuilt_problem.status not in {cp.OPTIMAL,cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Rebuilt flat-price problem: {rebuilt_problem.status}")
    rebuilt_power_copy=np.asarray(rebuilt_power.value).copy()
    rebuilt_bill=float(np.sum(flat_prices*rebuilt_power_copy*dt))
    print(f"Rebuilt flat-price bill: £{rebuilt_bill:.3f}")
    ''')
    task('D',r'''
    # Method 2: update the existing price Parameter.
    price.value=flat_prices.copy()
    flat_solution=solve_case(hard_problem)
    require_success(flat_solution,"Flat tariff hard comfort")
    tariff_solutions={"Flat":flat_solution,"ToU":hard_solution}
    cross_costs=pd.DataFrame({evaluation:{planning:float(np.dot(tariff,result["power_kw"])*dt)
        for planning,result in tariff_solutions.items()} for evaluation,tariff in tariffs.items()})
    cross_costs.index.name="optimisation_tariff";cross_costs.columns.name="evaluation_tariff"
    scheduling_value_tou=float(cross_costs.loc["Flat","ToU"]-cross_costs.loc["ToU","ToU"])
    ''',r'''
    # TODO D: assign flat_prices to price.value, solve hard_problem and copy flat_solution.
    # Retain hard_solution as the ToU plan in tariff_solutions.
    # Fill cross_costs: evaluate each saved power vector with each tariffs profile and dt.
    # scheduling_value_tou = ToU bill of Flat plan minus ToU bill of ToU plan.
    raise NotImplementedError("Task D: change the price Parameter and compare fixed schedules under common prices.")
    ''')
    code('tariff-figure',r'''
    tariff_energy=pd.DataFrame([{"optimisation_tariff":name,**schedule_metrics(result)}
        for name,result in tariff_solutions.items()])
    # Own-tariff summary: each bill uses the tariff that generated that schedule.
    tariff_summary=tariff_energy[["optimisation_tariff",*SUMMARY_COLUMNS]]
    display(tariff_summary.set_axis(["Optimise/evaluate under",*SUMMARY_LABELS],axis=1).round(3))
    export_table(tariff_summary,"tariff_summary.csv")
    # The cross-table answers a different question: hold the evaluation prices fixed.
    display(cross_costs.round(4))
    print(f"Value of changing schedule under ToU: £{scheduling_value_tou:.6f}")
    export_table(cross_costs,"tariff_cross_costs.csv",index=True);export_table(tariff_energy,"tariff_energy.csv")
    export_table(pd.DataFrame([{"scheduling_value_tou_gbp":scheduling_value_tou}]),"tariff_scheduling_value.csv")
    for name,result in tariff_solutions.items():export_schedule(result,"tariff_"+name.lower())
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for name,colour in [("Flat",COLOURS["flat"]),("ToU",COLOURS["tou"])]:
        result=tariff_solutions[name]
        axes[0].stairs(result["power_kw"],state_time,color=colour,label=name+" optimum")
        axes[1].plot(state_time,result["temperature_c"],color=colour,label=name+" optimum")
    axes[0].set(ylabel="Heat-pump electrical power (kW)",xlabel="Interval time")
    axes[0].legend(ncol=2,fontsize=10);comfort_lines(axes[1])
    axes[1].set(ylabel="Temperature (°C)",xlabel="Boundary time");axes[1].legend(fontsize=10)
    for ax in axes:clock_axis(ax)
    save_figure(fig,"F71-08",counts="2 hard plans; 48 powers and 49 states each")
    shown=cross_costs.reset_index().rename(columns={"optimisation_tariff":"Optimise under","Flat":"Evaluate at flat prices (£)","ToU":"Evaluate at ToU prices (£)"})
    export_table(shown,"F71-11_native_table.csv")
    fig=table_figure(shown,figsize=(12.5,3.5),fontsize=15)
    save_figure(fig,"F71-11",counts="2 optimisation tariffs x 2 evaluation tariffs")
    ''')
    md('tariff-interpretation',r'''
    ### Interpret tariffs under common evaluation prices (slides 29–31)

    Use one column of the bill table to isolate the effect of changing the schedule.
    The ToU optimum cannot cost more under ToU than the feasible flat plan, within
    solver tolerance; the corresponding inequality holds for flat evaluation.
    Report the computed energy and peak as well as bill and comfort. Positive flat
    prices make bill minimisation equivalent to energy minimisation. A ToU plan can
    use more electricity while costing less under ToU, but that pattern is not
    guaranteed. Neither strict savings nor unique schedules are assumed.

    The first summary table reports each schedule under its **own** tariff, as a
    description of those two plans. Its diagonal bills change both prices and
    schedule. Use the separate cross-table to isolate the value of changing the
    schedule under common prices.
    ''')
    md('task-e-text',r'''
    ## O5. Task E — lower comfort slack and its penalty (slide 36; 12 min)

    Introduce nonnegative slack $s_k$ to allow a lower-temperature shortfall.
    Following the lecture notation, the lower-soft problem is

    $$
    \begin{aligned}
    \min_{P,T,s}\quad & J_{\mathrm{bill}}(P)+\gamma\phi(s) \\
    \text{subject to}\quad & T_0=T_{\mathrm{init}}, \\
    & T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k,
        && \forall k\in\mathcal{K}, \\
    & 0\le P_k\le P_{\max}, && \forall k\in\mathcal{K}, \\
    & T_{k+1}+s_k\ge L_k, && \forall k\in\mathcal{K}, \\
    & T_{k+1}\le U_k, && \forall k\in\mathcal{K}, \\
    & s_k\ge 0, && \forall k\in\mathcal{K},
    \end{aligned}
    $$

    where $\phi(s)=\Delta t\sum_{k=0}^{N-1}s_k$. Only the lower comfort bound is
    relaxed; the initial state, dynamics, capacity and upper comfort bound remain hard.

    $\phi(s)$ is total slack in °C h; $\gamma$ has units £/(°C h). A 0.4°C
    shortfall for 0.5 h gives 0.2°C h; at £2.50/(°C h), the penalty is £0.50.
    The assigned penalty is separate from the metered electricity bill.

    Complete the lower-soft constraints and objective. The supplied sweep includes
    the lecture values 0.25, 2.5 and 25 plus intermediate values. Examine the bill,
    physical shortfall and penalised objective as $\gamma$ increases. LP solutions
    can change at thresholds and have plateaus: the response need not be smooth.
    A finite penalty does not guarantee zero shortfall in every problem.

    The lecture also mentions quadratic and absolute penalties. They require their
    own units/weight interpretation. With nonnegative slack, absolute slack equals
    slack; a quadratic penalty gives a different trade-off and a QP. This task uses
    only the total-slack definition above.
    ''')
    code('soft-parameters',r'''
    price.value=tou_prices.copy();initial_temperature.value=TRUE_T0
    slack=cp.Variable(N,nonneg=True,name="lower_shortfall_allowance_c")
    gamma=cp.Parameter(nonneg=True,value=.25,name="penalty_gbp_per_degree_hour")
    ''')
    task('E',r'''
    soft_constraints=base_constraints+[temperature[1:]+slack>=lower,temperature[1:]<=upper]
    phi=dt*cp.sum(slack)
    penalty_expression=gamma*phi
    soft_objective=cp.Minimize(bill_expression+penalty_expression)
    soft_problem=cp.Problem(soft_objective,soft_constraints)
    ''',r'''
    # TODO E: form soft_constraints from base_constraints, lower slack and hard upper bounds.
    # phi = dt * cp.sum(slack); penalty_expression = gamma * phi
    # soft_objective = cp.Minimize(...)
    # soft_problem = cp.Problem(soft_objective, soft_constraints)
    raise NotImplementedError("Task E: introduce lower slack and a dimensionally consistent penalised objective.")
    ''')
    code('soft-figure',r'''
    soft_solution=solve_case(soft_problem,hard=False,objective_kind="soft",slack_variable=slack,penalty_rate=float(gamma.value))
    require_success(soft_solution,"Lower-soft nominal case")
    comfort_comparison=pd.DataFrame([{"case":name,**schedule_metrics(result)}
        for name,result in [("Hard ToU",hard_solution),("Lower-soft gamma=0.25",soft_solution)]])
    display(comfort_comparison.round(4));export_table(comfort_comparison,"hard_soft_metrics.csv")
    export_schedule(soft_solution,"soft_tou")
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for name,result,colour in [("Hard",hard_solution,COLOURS["hard"]),("Lower-soft γ=0.25",soft_solution,COLOURS["soft"])]:
        axes[0].stairs(result["power_kw"],state_time,color=colour,label=name)
        axes[1].plot(state_time,result["temperature_c"],color=colour,label=name)
    axes[0].set(ylabel="Heat-pump electrical power (kW)",xlabel="Interval time");axes[0].legend(fontsize=10)
    comfort_lines(axes[1]);axes[1].set(ylabel="Temperature (°C)",xlabel="Boundary time");axes[1].legend(fontsize=10)
    for ax in axes:clock_axis(ax)
    save_figure(fig,"F71-05")
    ''')
    code('penalty-sweep',r'''
    penalty_solutions={};penalty_rows=[]
    PENALTY_WEIGHTS=[.25,.30,.35,.40,.50,.75,1.,1.5,2.5,5.,10.,25.]
    for weight in PENALTY_WEIGHTS:
        gamma.value=weight
        result=solve_case(soft_problem,hard=False,objective_kind="soft",slack_variable=slack,penalty_rate=weight)
        penalty_solutions[weight]=result
        row={"gamma_gbp_per_degree_hour":weight,"success":result["success"],"status":result["status"]}
        if result["success"]:
            row.update(schedule_metrics(result));export_schedule(result,f"penalty_{weight:g}")
        else:row["reason"]=result.get("reason","")
        penalty_rows.append(row)
    penalty_table=pd.DataFrame(penalty_rows);display(penalty_table.round(4));export_table(penalty_table,"penalty_sensitivity.csv")

    fig,ax=plt.subplots(figsize=(12.5,3.5))
    successful_penalties=penalty_table.loc[penalty_table.success]
    ax.plot(successful_penalties.below_band_degree_hours,successful_penalties.bill_gbp,"o-",color=COLOURS["soft"])
    for weight,offset in [(.25,(10,-12)),(2.5,(10,14)),(25.,(10,-20))]:
        row=successful_penalties.loc[successful_penalties.gamma_gbp_per_degree_hour.eq(weight)]
        if len(row):
            point=row.iloc[0]
            ax.annotate(f"γ={weight:g}",(point.below_band_degree_hours,point.bill_gbp),
                xytext=offset,textcoords="offset points")
    ax.scatter(hard_metrics["below_band_degree_hours"],hard_solution["bill_gbp"],marker="*",s=150,color=COLOURS["hard"],label="Hard-comfort baseline")
    ax.set(xlabel="Lower-temperature shortfall (°C h)",ylabel="Electricity bill (£)",title="Common ToU tariff and physical conditions")
    ax.margins(x=.18,y=.25);ax.legend(loc="upper right")
    save_figure(fig,"F71-06",counts=f"{len(PENALTY_WEIGHTS)} penalty cases plus hard baseline")

    # Show how the returned metrics change with the penalty weight.
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    axes[0].plot(successful_penalties.gamma_gbp_per_degree_hour,successful_penalties.bill_gbp,"o-",label="Electricity bill")
    axes[0].plot(successful_penalties.gamma_gbp_per_degree_hour,successful_penalties.total_objective_gbp,"s--",label="Bill + comfort penalty")
    axes[0].axhline(hard_solution["bill_gbp"],color="black",linestyle=":",label="Hard-comfort bill")
    axes[0].set_ylabel("Bill / penalised objective (£)");axes[0].legend(fontsize=9)
    axes[1].plot(successful_penalties.gamma_gbp_per_degree_hour,successful_penalties.below_band_degree_hours,"o-",color=COLOURS["soft"])
    axes[1].set_ylabel("Lower-temperature shortfall (°C h)")
    for ax in axes:
        ax.set(xscale="log",xlabel="γ (£ per °C h)")
        for row in penalty_rows:
            if not row["success"]:ax.text(row["gamma_gbp_per_degree_hour"],.9,"FAILED",transform=ax.get_xaxis_transform(),color="tab:red",ha="center")
    save_figure(fig,"F71-14",counts=f"{len(PENALTY_WEIGHTS)} fixed-case penalty values; failures retained")
    ''')
    md('penalty-interpretation',r'''
    The larger weights approach the hard-comfort bill in this case as lower
    shortfall is discouraged. Use the actual table/plots to identify thresholds or
    plateaus; do not interpret the weight as an empirically measured willingness
    to pay. If a solve fails, report its status and do not interpret missing values
    as a valid schedule. Increasing a weight does not change the equipment capacity.
    ''')
    md('task-f-text',r'''
    ## O6. Task F — minimise peak power (slide 40; 7 min)

    The peak-minimising schedule retains the same hard-comfort constraints:

    $$
    \begin{aligned}
    \min_{P,T,P_{\mathrm{peak}}}\quad & P_{\mathrm{peak}} \\
    \text{subject to}\quad & T_0=T_{\mathrm{init}}, \\
    & T_{k+1}=(1-b)T_k+bT_{o,k}+c\,\mathrm{COP}\,P_k,
        && \forall k\in\mathcal{K}, \\
    & 0\le P_k\le P_{\max}, && \forall k\in\mathcal{K}, \\
    & L_k\le T_{k+1}\le U_k, && \forall k\in\mathcal{K}, \\
    & P_k\le P_{\mathrm{peak}}, && \forall k\in\mathcal{K}, \\
    & P_{\mathrm{peak}}\ge 0.
    \end{aligned}
    $$

    Complete the constraint and objective; the solve, checks and plots are supplied.
    The objective is in **kW**, not £. Compare actual `max(power)`, kWh and the bill
    under the same ToU prices with the cost-optimal schedule. The minimum peak cannot
    exceed that of a feasible baseline beyond tolerance. Energy and bill need not
    fall. Several minimum-peak schedules may have different energy use and bills;
    we describe the returned optimum without an undisclosed secondary objective.
    ''')
    code('peak-parameter',r'''
    initial_temperature.value=TRUE_T0;price.value=tou_prices.copy()
    P_peak=cp.Variable(nonneg=True,name="heat_pump_peak_kw")
    ''')
    task('F',r'''
    peak_constraints=hard_constraints+[power<=P_peak]
    peak_objective=cp.Minimize(P_peak)
    peak_problem=cp.Problem(peak_objective,peak_constraints)
    ''',r'''
    # TODO F: add the power <= P_peak epigraph constraint to hard_constraints.
    # Minimise P_peak and construct peak_problem.
    raise NotImplementedError("Task F: minimise heat-pump peak power with unchanged hard-comfort physics.")
    ''')
    code('peak-comparison',r'''
    peak_solution=solve_case(peak_problem,objective_kind="peak",peak_variable=P_peak)
    require_success(peak_solution,"Minimum peak")
    peak_metrics=pd.DataFrame([{"case":name,**schedule_metrics(result)}
        for name,result in [("Cost optimum",hard_solution),("Peak optimum",peak_solution)]])
    peak_summary=peak_metrics[["case",*SUMMARY_COLUMNS]]
    display(peak_summary.set_axis(["Schedule",*SUMMARY_LABELS],axis=1).round(3))
    export_table(peak_summary,"peak_summary.csv")
    export_table(peak_metrics,"peak_comparison_metrics.csv")
    export_schedule(peak_solution,"peak_optimum")
    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for name,result,colour in [("Cost optimum",hard_solution,COLOURS["hard"]),("Peak optimum",peak_solution,COLOURS["peak"])]:
        axes[0].stairs(result["power_kw"],state_time,color=colour,label=name)
    axes[0].set(ylabel="Heat-pump electrical power (kW)",xlabel="Interval time");axes[0].legend(fontsize=10);clock_axis(axes[0])
    axes[1].axis("off")

    # The figure carries the same common metrics in a compact vertical table.
    rows=[[label,*[f"{value:.3f}" for value in peak_metrics[column]]]
        for column,label in zip(SUMMARY_COLUMNS,SUMMARY_LABELS)]
    tab=axes[1].table(cellText=rows,colLabels=["Metric","Cost optimum","Peak optimum"],
        cellLoc="center",loc="center",colWidths=[.48,.26,.26])
    tab.auto_set_font_size(False);tab.set_fontsize(8.5);tab.scale(1,1.6)
    axes[1].set_title("Common ToU evaluation; hard comfort",fontsize=11)
    save_figure(fig,"F71-12",counts="2 objectives; 48 powers and 49 states each")
    ''')
    md('interpretation',r'''
    ### Other objectives and the Week 8 boundary (slides 39–42)

    Energy $\Delta t\sum_kP_k$ is in kWh, bill $\Delta t\sum_kp_kP_k$ is in £,
    and peak power is in kW. Each ranks feasible schedules differently. Combining
    bill and peak requires a weight in £/kW; such a weight is not automatically a
    real demand charge. A comfort-tracking target also differs from an admissible band.

    Explain one tariff effect under common evaluation prices, one comfort trade-off,
    and one effect of the peak objective using the computed records. The supplied
    model, known-input assumption and endpoint constraints limit these conclusions.
    Today we re-solve complete plans for different parameters. Week 8 adds measurement,
    first-action application, horizon shifting and re-optimisation; this notebook
    contains no MPC feedback loop.
    ''')
    md('initial-error-text',r'''
    ## Optional: signed initial-temperature error (slide 44)

    This supplied extension is **not Task F** and lies outside the core timing.
    Plan using $T_0^{measured}=20+e_T$, with $e_T=[-1,-.5,0,.5,1]$°C. Replay every
    fixed schedule from the same true 20°C using identical physics and disturbances.
    No re-optimisation occurs during replay. Report lower and upper realised violations
    and failed cases; exclude the initial point from endpoint comfort scores. This
    isolates information error, not independent-plant validation or feedback control.
    ''')
    code('initial-error-case',r'''
    def simulate_temperature(electric_power_kw,initial_c=TRUE_T0):
        states=[float(initial_c)]
        for k,p in enumerate(electric_power_kw):
            states.append((1-b)*states[-1]+b*outdoor[k]+c*COP*p)
        return np.asarray(states)

    def initial_error_case(mode,error):
        measured_initial=TRUE_T0+error
        initial_temperature.value=measured_initial
        price.value=tou_prices.copy();gamma.value=.25
        problem=soft_problem if mode=="Soft" else hard_problem
        options={"hard":False,"objective_kind":"soft","slack_variable":slack,"penalty_rate":float(gamma.value)} if mode=="Soft" else {}
        result=solve_case(problem,**options)
        record={"mode":mode,"error_c":error,"measured_initial_c":measured_initial,"true_initial_c":TRUE_T0,
            "success":result["success"],"status":result["status"]}
        if not result["success"]:
            record["reason"]=result.get("reason","");return record,result,None
        realised=simulate_temperature(result["power_kw"],initial_c=TRUE_T0)
        realised_lower=np.maximum(lower-realised[1:],0.)
        realised_upper=np.maximum(realised[1:]-upper,0.)
        record.update(schedule_metrics(result))
        record.update(deviation_metrics(realised_lower,realised_upper))
        record["true_initial_c"]=TRUE_T0
        return record,result,realised
    ''')
    code('error-sweep',r'''
    error_records=[];error_solutions={};error_replays={}
    for mode in ["Hard","Soft"]:
        for error in [-1.,-.5,0.,.5,1.]:
            record,result,realised=initial_error_case(mode,error)
            error_records.append(record);error_solutions[mode,error]=result;error_replays[mode,error]=realised
            if result["success"]:
                stem=f"initial_error_{mode.lower()}_{error:+g}"
                export_schedule(result,stem)
                export_table(pd.DataFrame({"state_time":state_time,"true_replay_temperature_c":realised}),stem+"_replay.csv")
    error_table=pd.DataFrame(error_records);display(error_table.round(4));export_table(error_table,"initial_error_metrics.csv")

    fig,axes=plt.subplots(1,2,figsize=(12.5,3.5))
    for mode,colour in [("Hard",COLOURS["hard"]),("Soft",COLOURS["soft"])]:
        group=error_table.loc[error_table["mode"].eq(mode)].sort_values("error_c")
        axes[0].plot(group.error_c,group.get("bill_gbp",pd.Series(np.nan,index=group.index)),"o-",color=colour,label=mode)
        for column,style,label in [("below_band_degree_hours","o-","lower"),("above_band_degree_hours","s--","upper")]:
            axes[1].plot(group.error_c,group.get(column,pd.Series(np.nan,index=group.index)),style,color=colour,label=mode+" "+label)
        for ax in axes:
            for error in group.loc[~group.success,"error_c"]:
                ax.text(error,.85 if mode=="Hard" else .7,mode+" FAILED",transform=ax.get_xaxis_transform(),color=colour,ha="center",fontsize=10)
            ax.axvline(0,color="0.5",lw=.8);ax.set_xlabel("Planning observation − true initial T (°C)")
    axes[0].set_ylabel("Electricity bill (£)");axes[1].set_ylabel("Realised shortfall / excess (°C h)")
    axes[0].legend(fontsize=10);axes[1].legend(fontsize=9,ncol=2)
    save_figure(fig,"F71-09",counts="10 planned cases: 2 comfort policies x 5 signed errors; failures retained")
    ''')
    md('error-interpretation',r'''
    Read positive error as an initially warm reading. Compare both signs using both
    lower and upper realised discomfort alongside bill and energy. A hard predicted
    bound need not hold in true-state replay; no error sign must worsen every outcome.
    The common true initial state and fixed actions are retained throughout replay.
    ''')
    md('optional',r'''
    ## Optional: tracking and terminal treatment (slides 45–46)

    **Quadratic tracking:** $J_{track}=\Delta t\sum_k(T_{k+1}-T_{ref,k})^2$ has units
    °C² h. Nonnegative weights on squared affine terms preserve convexity; with affine
    constraints this gives a convex QP. A weight in £/(°C² h) can combine tracking with
    bill. A target is distinct from a permitted band. No further core task is required.

    **Terminal treatment:** the core applies comfort at the final boundary without
    assigning value to later stored heat. A longer horizon, terminal cost or extra
    target changes the formulation and may change feasibility. Hold terminal treatment
    common across comparisons; do not quietly add $T_N=T_0$.

    An optional secondary optimisation among minimum-peak schedules must declare its
    priority and tolerance explicitly. No tie-breaking objective is used here.
    A separate independent-plant comparison would require a declared plant, common
    realised inputs and initial states; it is not implemented in this session.
    ''')
    code('manifest',r'''
    if EXPORT_ARTEFACTS:
        pd.DataFrame(figure_manifest).sort_values("id").to_csv(ROOT/"teaching/week7_day1_figure_manifest.csv",index=False)
        provenance={"sources":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_paths},
            "period_start":str(interval_start[0]),"period_end":str(interval_end[-1]),"N":N,"dt_hours":dt,
            "common_model_role":"Instructor model fitted on all 28 days, not the Week 6 held-out student fit",
            "comfort":"Interval requirements checked at next endpoint; T0 separate; no within-interval guarantee",
            "terminal_policy":"Final comfort only; no additional equality","information":"Known future inputs for deterministic planning",
            "solver":SOLVER,"options":SOLVER_OPTIONS,"feasibility_tolerance":FEASIBILITY_TOL,
            "comfort_count_tolerance_c":COMFORT_COUNT_TOL,"raw_solver_values":"Retained without clipping"}
        (OUTPUT_DIR/"model_and_case_provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf8")
    ''')
    md('references',r'''
    ### References and source credits (slide 43)

    - [Reference data/model README](data/reference_building/README.md): supplied synthetic teaching case.
    - Boyd, S. and Vandenberghe, L. (2004), *Convex Optimization*, Cambridge University Press.
    - Diamond, S. and Boyd, S. (2016), *CVXPY: A Python-Embedded Modeling Language for Convex Optimization*, JMLR 17(83), 1–5.
    - [CVXPY DCP guide](https://www.cvxpy.org/tutorial/dcp/index.html) and
      [solver/status documentation](https://www.cvxpy.org/tutorial/solvers/index.html).
    - Drgoňa et al. (2020), *All you need to know about model predictive control for buildings*, Annual Reviews in Control 50, 190–232.
    ''')
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})


def build():
    for solved in [False,True]:nbf.write(make_notebook(solved),ROOT/(STEM+('_solution' if solved else '')+'.ipynb'))
    teaching=ROOT/'teaching';teaching.mkdir(exist_ok=True)
    with (teaching/'week7_day1_task_manifest.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['task_id','slide','minutes_including_debrief','cell_id'])
        for letter,slide,minutes in TASKS:writer.writerow([letter,slide,minutes,'w71-task-'+letter.lower()])
    with (teaching/'week7_day1_schedule.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.writer(stream);writer.writerow(['slide','core_minutes','block'])
        for slide,minutes in enumerate(SLIDE_MINUTES,1):writer.writerow([slide,minutes,1 if slide<=11 else 2 if slide<=21 else 3 if slide<=31 else 4])
    print('Built Tutorial 7.1 pair only: six tasks, 62 task minutes within 200 teaching minutes; optional topics separate.')


if __name__=='__main__':build()
