"""Scoped authoring for Tutorial 6.1 only. Reconcile live manual edits first.

Original pasted notebooks were inspected and backed up outside the repository.
This script never calls the older Weeks 6--8 builder.
"""
from pathlib import Path
import csv
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
STEM = 'tutorial_6_1_control_oriented_building_model'
FIGURES = [
    ('F61-01', 'w6d1_F01_operating_conditions', '4', 'M0', 'Full-width', '96 intervals; 48 h',
     'Synthetic outdoor temperature, supplied comfort bounds and prescribed useful heat.'),
    ('F61-02', 'w6d1_F02_zone_heat_flows', '5', 'M0', 'Right-hand', '',
     'Zone boundary, storage, outdoor exchange and heat/electricity distinction.'),
    ('F61-03', 'w6d1_F03_one_r_one_c', '11', 'M2', 'Centred wide', '',
     'One resistance and one thermal capacitance; gains omitted in the compact core.'),
    ('F61-04', 'w6d1_F04_transition_timing', '13;19', 'M2-M3', 'Full-width', '',
     'Two held-input intervals connect three boundary states.'),
    ('F61-05', 'w6d1_F05_schedule_rollout', '22', 'M3', 'Full-width', '96 intervals; 97 states; 48 h',
     'Scheduled and unheated compact simulations share weather and initial temperature.'),
    ('F61-06', 'w6d1_F06_resistance_sensitivity', '24', 'M4', 'Full-width', '192 intervals; 193 states per case; 96 h',
     'Resistance changes equilibrium and time constant; final plotted values are not equilibria.'),
    ('F61-07', 'w6d1_F07_capacitance_sensitivity', '25', 'M4', 'Full-width', '192 intervals; 193 states per case; 96 h',
     'Capacitance changes response speed with the same 32 degree C equilibrium.'),
    ('F61-08', 'w6d1_F08_time_step_error', '29', 'M5', 'Full-width', '192;24;4 intervals; 193;25;5 states; 96 h',
     'Euler errors use each method\'s actual times; the 24 h method is oscillatory and stable.'),
    ('F61-09', 'w6d1_F09_reference_plant', '30', 'M6', 'Full-width', '',
     'Supplied fabric/air topology, ventilation, gain allocations and measured-air channel.'),
    ('F61-10', 'w6d1_F10_model_comparison', '32', 'M6', 'Full-width', '96 intervals; 97 states per model; 48 h',
     'Noiseless plant and compact model share realised useful heat, weather and initial air state; only plant has gains.'),
]
TASKS = [
    ('A', 'M1', 9, 10, 'Useful heat power and interval energies'),
    ('B', 'M2', 14, 12, 'Scalar thermal transition'),
    ('C', 'M3', 21, 10, 'All N transitions and N+1 states'),
    ('D', 'M4', 23, 10, 'Sweep values and common-case simulation calls'),
    ('E', 'M5', 28, 8, 'Euler step list and time-aligned error'),
    ('F', 'M6', 31, 12, 'Compact simulation under recorded common inputs'),
]


def continuation(head, tail, indent=0):
    return (textwrap.dedent(head).strip() + '\n' +
            textwrap.indent(textwrap.dedent(tail).strip(), ' ' * indent))


def make_notebook(solution=False):
    cells = []

    def md(key, source):
        cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip(), id=key))

    def code(key, source, task=None):
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip(), id=key,
                     metadata={'tags': ['task', task]} if task else {}))

    def task(letter, answer, scaffold):
        code('w61-task-' + letter.lower(), answer if solution else scaffold, letter)

    md('w61-purpose', r'''
    # Tutorial 6.1: Control-oriented building modelling and simulation

    How quickly does a room cool when its heater switches off? Use a compact
    thermal model to simulate the response to proposed inputs, then compare it
    with a richer, synthetic reference plant.

    **Outcomes:** distinguish power and energy; implement one thermal transition
    and a free rollout; explain resistance, capacitance and time-step effects;
    specify the model's inputs, timing and limitations.

    | Block | Slides | Minutes | Focus |
    |---|---|---:|---|
    | 1 | 1–10 | 50 | Operating question, heat flows and Task A |
    | 2 | 11–18 | 50 | One-step dynamics and Task B |
    | Break | after 18 | 15–20 extra | Save and pause |
    | 3 | 19–26 | 50 | Rollouts and Tasks C–D |
    | 4 | 27–34 | 50 | Numerical error, mismatch and Tasks E–F |

    The 200 teaching minutes include coding and debriefs. Optional self-study is
    outside this total. Plotting is supplied after the relevant calculation.
    Run from the repository root in the module's pinned environment.
    ''')
    md('w61-m0', r'''
    ## M0. Inputs, units and the running example (slides 4–8)

    The supplied [reference data](data/reference_building/README.md) describe a
    synthetic single-zone dwelling. Inspect the
    first 96 rows of `operation_conditions.csv`: half-hour intervals over
    3–4 February 2025. Each row labels an interval **start**. The full operation
    file covers 14 days; the separate identification file contains 28 days of
    noisy synthetic measurements for 6.2.

    Core units: useful heat $q_{th}$ and electricity $P_{el}$ in kW; time in h;
    resistance $R$ in K/kW; capacitance $C$ in kWh/K; temperatures in °C.

    The compact state is indoor temperature. Outdoor temperature is a disturbance;
    useful delivered heat is an input. Solar/internal gains are **omitted** in the
    compact core. Comfort bounds are supplied requirements (not measurements).
    Weather is a realised simulation input here, not a perfect operational forecast.
    ''')
    code('w61-setup', '''
    from pathlib import Path
    import hashlib
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import FancyBboxPatch
    from IPython.display import display

    ROOT = Path.cwd()
    DATA_PATH = ROOT / "data/reference_building/operation_conditions.csv"
    if not DATA_PATH.is_file() or not (ROOT / "course_utils/reference_building.py").is_file():
        raise FileNotFoundError("Run from the repository root containing data/ and course_utils/.")
    from course_utils.reference_building import TwoStateReferencePlant, ReferenceBuildingParameters

    FIGURE_DIR = ROOT / "figures/week6_day1"
    OUTPUT_DIR = ROOT / "teaching/week6_day1_outputs"
    EXPORT_ARTEFACTS = True
    plt.rcParams.update({"figure.dpi": 110, "font.size": 12, "axes.grid": True,
                         "grid.alpha": .2, "svg.fonttype": "none", "savefig.facecolor": "white"})
    COLOURS = {"compact": "tab:blue", "reference": "black", "heat": "tab:orange",
               "outdoor": "0.5", "fabric": "tab:purple", "electric": "tab:green"}
    step_hours, controller_resistance, controller_capacitance = .5, 5.4, 4.0
    ''')
    md('w61-plotting', '''
    ### Supplied figure and export helpers

    These helpers style diagrams and export current calculations. They do not
    implement the thermal transition or recurrence. Set `EXPORT_ARTEFACTS=False`
    to run without writing files; no cell reads previously exported answers.
    ''')
    code('w61-export', 'FIGURE_SPECS = ' + repr(FIGURES) + '\n' + textwrap.dedent('''
    figure_manifest = []
    def save_figure(fig, figure_id, source_files=(), period="", units="", case="", action=""):
        spec = next(row for row in FIGURE_SPECS if row[0] == figure_id)
        fig.tight_layout()
        if EXPORT_ARTEFACTS:
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for extension in ["png", "svg"]:
                fig.savefig(FIGURE_DIR / f"{spec[1]}.{extension}", dpi=220, bbox_inches="tight")
            is_diagram = figure_id in {"F61-02", "F61-03", "F61-04", "F61-09"}
            if not source_files and not is_diagram:
                source_files = ("scripts/build_week6_day1.py",)
            source = "instructional diagram" if is_diagram else ";".join(source_files)
            hashes = ";".join(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files)
            figure_manifest.append({"id": figure_id, "png": f"figures/week6_day1/{spec[1]}.png",
                "svg": f"figures/week6_day1/{spec[1]}.svg", "slide": spec[2], "section": spec[3],
                "placement": spec[4], "source": source, "source_files": ";".join(source_files), "source_sha256": hashes, "period": period,
                "units": units, "sample_transition_count": spec[5], "model_case": case,
                "student_action": action or "Supplied", "caption": spec[6]})
        plt.show()
        plt.close(fig)

    def export_table(frame, filename, index=True):
        if EXPORT_ARTEFACTS:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            frame.to_csv(OUTPUT_DIR / filename, index=index)

    def time_axes(axes):
        for ax in np.atleast_1d(axes):
            ax.set_xlim(boundary_index[0], boundary_index[-1])
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\\n%H:%M"))
            ax.legend(fontsize=10, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3)
            if ax.get_title():
                ax.set_title(ax.get_title(), pad=30)

    def diagram(figsize=(10, 4)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(xlim=(0, 10), ylim=(0, 6)); ax.axis("off")
        return fig, ax

    def box(ax, x, y, width, height, text, colour="0.3", size=13):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.08",
                     linewidth=1.5, edgecolor=colour, facecolor="white"))
        ax.text(x+width/2, y+height/2, text, ha="center", va="center", fontsize=size)

    def arrow(ax, start, end, label="", colour="black", style="-", label_at=None):
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": colour,
                    "linewidth": 1.8, "linestyle": style})
        if label:
            x, y = label_at or ((start[0]+end[0])/2, (start[1]+end[1])/2+.28)
            ax.text(x, y, label, ha="center", va="center", fontsize=12, color=colour,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ''').strip())
    code('w61-inputs', '''
    conditions = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True).iloc[:96].copy()
    interval_index = conditions.index
    boundary_index = pd.date_range(interval_index[0], periods=len(conditions)+1, freq="30min", name="state_time")
    heat_schedule = pd.Series(0.0, index=interval_index, name="q_th_kw")
    heat_schedule.loc[(interval_index.hour >= 5) & (interval_index.hour < 8)] = 6.0
    heat_schedule.loc[(interval_index.hour >= 17) & (interval_index.hour < 22)] = 4.0
    operating_inputs = conditions.copy()
    operating_inputs.index.name = "interval_start"
    operating_inputs.insert(0, "interval_end", interval_index + pd.Timedelta(hours=step_hours))
    operating_inputs["q_th_kw"] = heat_schedule
    operating_inputs["dt_hours"] = step_hours
    display(operating_inputs.head())
    export_table(operating_inputs, "operating_inputs.csv")
    ''')
    code('w61-f01', '''
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[2, 1])
    for column, label, colour, style in [
        ("outdoor_temperature_c", "Outdoor (interval input)", COLOURS["outdoor"], "-"),
        ("comfort_min_c", "Supplied lower comfort limit", "tab:blue", "--"),
        ("comfort_max_c", "Supplied upper comfort limit", "tab:red", "--")]:
        axes[0].stairs(conditions[column], boundary_index, label=label, color=colour, linestyle=style)
    axes[0].set(ylabel="Temperature (°C)", title="Synthetic operation: 96 half-hour intervals / 48 h")
    axes[1].stairs(heat_schedule, boundary_index, color=COLOURS["heat"], label="Prescribed useful heat")
    axes[1].set(ylabel="Useful heat (kW)", xlabel="Interval time")
    time_axes(axes)
    save_figure(fig, "F61-01", ["data/reference_building/operation_conditions.csv"],
                str(boundary_index[0])+" to "+str(boundary_index[-1]), "°C;kW;h", "Prescribed schedule")
    ''')
    code('w61-f02', r'''
    fig, ax = diagram((8, 5))
    box(ax, 3.5, 1.65, 3.5, 2.9, "Zone boundary\nIndoor state T\nThermal storage C")
    box(ax, .1, 2.9, 1.55, .9, "Outdoor\n$T_o$")
    arrow(ax, (1.8, 3.35), (3.35, 3.35), "Exchange", label_at=(2.5, 3.9))
    box(ax, .1, .2, 2.3, 1.1, "Electricity $P_{el}$\nHeat pump / COP")
    arrow(ax, (2.5, .75), (4.1, 1.55), "$q_{th}$", COLOURS["heat"], label_at=(3.2, .8))
    arrow(ax, (8.7, 3.1), (7.15, 3.1), "$q_{gain}$", label_at=(8.2, 3.6))
    ax.text(8.1, 4.35, "Solar / internal", ha="center", fontsize=12)
    ax.text(5, 5.35, "Heat flows are signed positive into the zone", ha="center", fontsize=13)
    ax.text(5.1, .2, "$q_{th}=COP\\,P_{el}$; heat enters storage", ha="center", fontsize=12)
    save_figure(fig, "F61-02", units="kW;kWh/K;°C", case="Instructional energy boundary")
    ''')
    md('w61-m1', r'''
    ## M1. Task A — useful heat, power and energy (slide 9; 10 min)

    A heat pump draws $P_{el}=2$ kW with $COP=3$ for $\Delta t=0.5$ h.
    Complete useful heat power, electrical energy and delivered heat energy.
    Use $q_{th}=COP\,P_{el}$ and energy = power × interval duration.
    Which quantity enters the thermal balance? Write one sentence after the table.
    ''')
    task('A', '''
    P_el, COP, dt = 2.0, 3.0, .5
    q_th = COP * P_el
    E_el = P_el * dt
    E_th = q_th * dt
    ''', '''
    P_el, COP, dt = 2.0, 3.0, .5
    # TODO A: calculate q_th (kW), E_el (kWh) and E_th (kWh).
    raise NotImplementedError("Task A: complete the three power/energy expressions.")
    ''')
    code('w61-a-results', '''
    energy_example = pd.DataFrame({"Quantity": ["Useful heat power", "Electrical energy", "Delivered heat energy"],
                                   "Value": [q_th, E_el, E_th], "Unit": ["kW", "kWh", "kWh"]})
    display(energy_example)
    export_table(energy_example, "task_a_energy.csv", index=False)
    ''')
    md('w61-m2', r'''
    ## M2. One thermal transition (slides 11–18)

    One resistance represents outdoor heat exchange; one capacitance represents
    effective thermal storage. For the compact core, gains are omitted:
    $C\,dT/dt=(T_o-T)/R+q_{th}$. Positive useful heat warms the zone.

    $T_k$ belongs to boundary $t_k$. Hold inputs constant on
    $[t_k,t_{k+1})$ and predict $T_{k+1}$; do not label it at $t_k$.
    ''')
    code('w61-f03', r'''
    fig, ax = diagram((12, 4))
    ax.text(.6, 3.3, "$T_o$", fontsize=20)
    ax.plot([1.1, 2.4], [3.5, 3.5], color="black")
    box(ax, 2.4, 3.1, 1.2, .8, "$R$")
    ax.plot([3.7, 5.7], [3.5, 3.5], color="black")
    ax.plot(5.7, 3.5, "o", color="tab:blue", ms=9)
    ax.text(5.9, 3.7, "$T$", fontsize=20)
    ax.plot([5.7, 5.7], [3.4, 2.2], color="black")
    ax.plot([5.1, 6.3], [2.2, 2.2], color="black", lw=2)
    ax.plot([5.1, 6.3], [1.9, 1.9], color="black", lw=2)
    ax.plot([5.7, 5.7], [1.9, 1.0], color="black")
    ax.plot([5.1, 6.3], [1, 1], color="black")
    ax.text(6.6, 2.0, "$C$", fontsize=18)
    ax.text(5.7, .45, "Thermal reference", ha="center", fontsize=12)
    arrow(ax, (8.1, 3.5), (6.1, 3.5), "$q_{th}$", COLOURS["heat"], label_at=(7.2, 4.0))
    arrow(ax, (5.7, 5.1), (5.7, 3.9), "Heat gains omitted", "0.45", "--", (5.7, 5.5))
    ax.text(1.8, .7, "R: K/kW\nC: kWh/K\nHeat: kW", ha="center", fontsize=12)
    save_figure(fig, "F61-03", units="K/kW;kWh/K;kW;°C", case="Compact 1R1C")
    ''')
    code('w61-f04', r'''
    fig, ax = diagram((12, 3.6))
    for k, x in enumerate([1.3, 5, 8.7]):
        ax.plot(x, 3.8, "o", color=COLOURS["compact"], ms=12)
        suffix = "k" if k == 0 else f"k+{k}"
        ax.text(x, 4.5, "$T_{"+suffix+"}$", ha="center", fontsize=18)
        ax.text(x, 3.05, "$t_{"+suffix+"}$", ha="center", fontsize=16)
    ax.plot([1.3, 8.7], [3.8, 3.8], color="black", zorder=0)
    box(ax, 1.4, 1.15, 3.4, 1.0, "Held inputs $T_{o,k}, q_{th,k}$")
    box(ax, 5.1, 1.15, 3.4, 1.0, "Held inputs $T_{o,k+1}, q_{th,k+1}$")
    ax.text(5, .35, "N = 2 input intervals; N + 1 = 3 boundary states", ha="center", fontsize=15)
    save_figure(fig, "F61-04", units="h;°C;kW", case="Interval/boundary timing")
    ''')
    md('w61-task-b-text', r'''
    ### Task B — implement one transition (slide 14; 12 min)

    Use forward Euler: temperature change is $\Delta t/C$ times net useful heat.
    Complete the outdoor exchange, net heat and return expression in the short
    function below. Predict the direction for the supplied case before running it.
    ''')
    signature = '''
    def one_r_one_c_step(indoor_temperature_c, outdoor_temperature_c, delivered_heat_kw,
                         resistance_k_per_kw, capacitance_kwh_per_k, step_hours):
    '''
    task('B', continuation(signature, '''
        outdoor_exchange_kw = (outdoor_temperature_c - indoor_temperature_c) / resistance_k_per_kw
        net_heat_kw = outdoor_exchange_kw + delivered_heat_kw
        return indoor_temperature_c + step_hours * net_heat_kw / capacitance_kwh_per_k
    ''', 4), continuation(signature, '''
        # TODO B: outdoor exchange (positive inward), net heat, next temperature.
        raise NotImplementedError("Task B: implement the scalar thermal update.")
    ''', 4))
    code('w61-b-results', '''
    example_next_temperature = one_r_one_c_step(20, 5, 4, 5.4, 4, .5)
    balanced_heat_kw = (20-5)/5.4
    print(f"Example next temperature: {example_next_temperature:.6f} °C")
    print("No heat / balanced heat:", one_r_one_c_step(20,5,0,5.4,4,.5),
          one_r_one_c_step(20,5,balanced_heat_kw,5.4,4,.5))
    ''')
    md('w61-coefficients', r'''
    The same equation is $T_{k+1}=aT_k+bT_{o,k}+cq_{th,k}$, with
    $b=\Delta t/(RC)$, $c=\Delta t/C$ and $a=1-b$.
    $a,b$ are dimensionless; $c$ has units K/kW. For electrical power as input,
    use $c\,COP$, not $c$ alone.

    For positive $R,C$, $\tau=RC$. Euler is asymptotically stable when
    $0<\Delta t/\tau<2$. Non-oscillatory relaxation requires
    $0<\Delta t/\tau\leq1$. Neither condition establishes numerical accuracy.

    **Checkpoint:** identify the state, input and disturbance; explain why the
    model needs a time interval. **Break: 15–20 minutes after slide 18.**
    ''')
    md('w61-m3', r'''
    ## M3. Task C — free rollout (slide 21; 10 min)

    Initialise once and feed each model prediction into the next update. Use all
    N input intervals to produce N+1 boundary states. Unlike one-step prediction,
    a free rollout does not reset from later temperature measurements.

    Complete allocation, initial condition and recurrence. Keep the supplied index
    construction: 96 inputs yield 97 states, ending at 5 February 00:00, exactly
    48 h after the first boundary. The final input must affect the final state.
    ''')
    sim_header = '''
    def simulate_one_r_one_c(outdoor_temperature_c, delivered_heat_kw, initial_temperature_c,
                             resistance_k_per_kw, capacitance_kwh_per_k, step_hours):
        if not outdoor_temperature_c.index.equals(delivered_heat_kw.index):
            raise ValueError("Outdoor temperature and heat must have matching interval starts.")
        state_index = pd.date_range(outdoor_temperature_c.index[0],
                                   periods=len(outdoor_temperature_c)+1,
                                   freq=pd.Timedelta(hours=step_hours), name="state_time")
    '''
    task('C', continuation(sim_header, '''
        temperatures = np.empty(len(outdoor_temperature_c)+1)
        temperatures[0] = initial_temperature_c
        for k, (outdoor, heat) in enumerate(zip(outdoor_temperature_c, delivered_heat_kw)):
            temperatures[k+1] = one_r_one_c_step(temperatures[k], outdoor, heat,
                resistance_k_per_kw, capacitance_kwh_per_k, step_hours)
        return pd.Series(temperatures, index=state_index, name="indoor_temperature_c")
    ''', 4), continuation(sim_header, '''
        # TODO C: allocate N+1 temperatures, set T_0, and update all N intervals.
        # Return a Series with state_index; each update uses your previous prediction.
        raise NotImplementedError("Task C: return all N+1 boundary states.")
    ''', 4))
    code('w61-rollouts', '''
    no_heat = pd.Series(0.0, index=interval_index)
    temperature_no_heat = simulate_one_r_one_c(conditions.outdoor_temperature_c, no_heat,
                            20.0, controller_resistance, controller_capacitance, step_hours)
    temperature_scheduled = simulate_one_r_one_c(conditions.outdoor_temperature_c, heat_schedule,
                            20.0, controller_resistance, controller_capacitance, step_hours)
    rollout_states = pd.DataFrame({"no_heat_c": temperature_no_heat, "scheduled_c": temperature_scheduled})
    export_table(rollout_states, "schedule_states.csv")
    fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, height_ratios=[2, 1])
    axes[0].plot(rollout_states.index, rollout_states.no_heat_c, color=COLOURS["compact"], label="No heat")
    axes[0].plot(rollout_states.index, rollout_states.scheduled_c, color=COLOURS["heat"], label="Scheduled")
    axes[0].stairs(conditions.outdoor_temperature_c, boundary_index, color=COLOURS["outdoor"], label="Outdoor (interval input)")
    axes[0].set(ylabel="Temperature (°C)", title="Same weather and T₀ = 20°C; 96 inputs → 97 states")
    axes[1].stairs(heat_schedule, boundary_index, color=COLOURS["heat"], label="Useful heat schedule")
    axes[1].set(ylabel="Useful heat (kW)", xlabel="Boundary / interval time")
    time_axes(axes)
    save_figure(fig, "F61-05", ["data/reference_building/operation_conditions.csv"],
                str(boundary_index[0])+" to "+str(boundary_index[-1]), "°C;kW;h", "1R1C; R=5.4; C=4", "Task C recurrence")
    ''')
    md('w61-rollout-interpret', '''
    **Interpret:** compare the temperatures at the same boundary time. Why can
    temperature fall while heating is on? Is this a comfort guarantee or an
    optimised schedule? Use the plotted heat input and outdoor temperature.
    ''')
    md('w61-m4', r'''
    ## M4. Task D — resistance and capacitance experiments (slide 23; 10 min)

    Predict first. Hold outdoor temperature at 5°C, useful heat at 5 kW and
    initial temperature at 18°C for 96 h (192 half-hour intervals, 193 states).
    Vary $R=[3.5,5.4,8]$ with $C=4$; then vary $C=[2,4,8]$ with $R=5.4$.
    Complete the lists and simulation calls in the supplied loop structure.
    The constant-input equilibrium is $T_{eq}=T_o+Rq_{th}$ and $\tau=RC$.
    ''')
    d_start = '''
    experiment_index = pd.date_range("2025-01-01", periods=192, freq="30min")
    experiment_outdoor = pd.Series(5.0, index=experiment_index)
    experiment_heat = pd.Series(5.0, index=experiment_index)
    resistance_responses, capacitance_responses = {}, {}
    '''
    task('D', continuation(d_start, '''
    resistance_values, capacitance_values = [3.5, 5.4, 8.0], [2.0, 4.0, 8.0]
    for resistance in resistance_values:
        resistance_responses[resistance] = simulate_one_r_one_c(
            experiment_outdoor, experiment_heat, 18.0, resistance, 4.0, .5)
    for capacitance in capacitance_values:
        capacitance_responses[capacitance] = simulate_one_r_one_c(
            experiment_outdoor, experiment_heat, 18.0, 5.4, capacitance, .5)
    ''', 0), continuation(d_start, '''
    # TODO D: complete the two lists and the two simulation calls below.
    raise NotImplementedError("Task D: set sweep values and vary one parameter at a time.")
    resistance_values, capacitance_values = [], []
    for resistance in resistance_values:
        resistance_responses[resistance] = None  # Replace with your simulation call.
    for capacitance in capacitance_values:
        capacitance_responses[capacitance] = None  # Replace with your simulation call.
    ''', 0))
    code('w61-sweep-results', '''
    parameter_rows, parameter_states = [], []
    for kind, responses, fixed in [("R", resistance_responses, 4.0), ("C", capacitance_responses, 5.4)]:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        for value, response in responses.items():
            R, C = (value, fixed) if kind == "R" else (fixed, value)
            equilibrium, tau = 5+R*5, R*C
            hours = (response.index-response.index[0]).total_seconds()/3600
            line, = ax.plot(hours, response, label=f"{kind}={value:g}; τ={tau:g} h")
            if kind == "R":
                ax.axhline(equilibrium, color=line.get_color(), linestyle="--", alpha=.7,
                           label=f"Equilibrium {equilibrium:g}°C")
            parameter_rows.append({"sweep": kind, "R_K_per_kW": R, "C_kWh_per_K": C,
                                   "T_eq_c": equilibrium, "tau_hours": tau, "T_96h_c": response.iloc[-1]})
            parameter_states.append(pd.DataFrame({"sweep": kind, "value": value, "state_time": response.index,
                                                  "elapsed_hours": hours, "temperature_c": response.to_numpy()}))
        if kind == "C": ax.axhline(32, color="black", linestyle="--", label="Common equilibrium 32°C")
        ax.set(xlabel="Elapsed time (h)", ylabel="Indoor temperature (°C)", xlim=(0,96), ylim=(17,47),
               title="Resistance: C = 4 kWh/K" if kind == "R" else "Capacitance: R = 5.4 K/kW")
        ax.legend(ncol=3, fontsize=11, loc="lower left", bbox_to_anchor=(0, 1.01))
        ax.set_title(ax.get_title(), pad=48)
        save_figure(fig, "F61-06" if kind == "R" else "F61-07", period="Idealised 96 h experiment",
                    units="°C;K/kW;kWh/K;h", case=f"{kind} sweep; To=5°C; q=5 kW; T0=18°C", action="Task D calls")
    parameter_table = pd.DataFrame(parameter_rows)
    display(parameter_table.round(4))
    export_table(parameter_table, "parameter_sweep.csv", index=False)
    export_table(pd.concat(parameter_states, ignore_index=True), "parameter_states.csv", index=False)
    ''')
    md('w61-sweep-interpret', r'''
    **Interpret:** $R$ changes both equilibrium and time constant. $C$ changes
    response speed with a common equilibrium. Compare final values with the dashed
    equilibrium lines; 96 h is not a claim that every curve has fully converged.
    Similar short responses can arise from different parameter combinations.
    ''')
    md('w61-m5', r'''
    ## M5. Task E — sampling interval and numerical error (slide 28; 8 min)

    Keep the same 96 h physical case, $R=5.4$, $C=4$, $T_o=5$, $q=5$, $T_0=18$.
    The analytic constant-input solution is supplied below. Complete the Euler
    step list (0.5, 4, 24 h) and error expression. Evaluate the analytic reference
    at **each Euler trajectory's own times**, without interpolating trajectories.
    ''')
    code('w61-analytic', '''
    def exact_constant_input(hours, T0, T_out, q_th, R, C):
        equilibrium = T_out + R*q_th
        return equilibrium + (T0-equilibrium)*np.exp(-np.asarray(hours)/(R*C))
    ''')
    e_setup = '''
    euler_results, euler_rows = {}, []
    '''
    task('E', continuation(e_setup, '''
    time_steps = [.5, 4.0, 24.0]
    for dt in time_steps:
        times = pd.date_range("2025-01-01", periods=int(96/dt), freq=pd.Timedelta(hours=dt))
        response = simulate_one_r_one_c(pd.Series(5.,index=times), pd.Series(5.,index=times), 18.,5.4,4.,dt)
        hours = np.arange(len(response))*dt
        exact_at_times = exact_constant_input(hours,18.,5.,5.,5.4,4.)
        error = response.to_numpy()-exact_at_times
        euler_results[dt] = pd.DataFrame({"state_time":response.index,"elapsed_hours":hours,
            "euler_c":response.to_numpy(),"exact_c":exact_at_times,"error_c":error})
        ratio = dt/(5.4*4)
        euler_rows.append({"dt_hours":dt,"a":1-ratio,"dt_over_tau":ratio,
                           "max_abs_error_c":np.max(np.abs(error)),"N_intervals":len(times),"N_states":len(response)})
    ''', 0), continuation(e_setup, '''
    # TODO E: set time_steps and replace the error expression in the supplied loop.
    raise NotImplementedError("Task E: choose 0.5, 4 and 24 h and compare errors at matching times.")
    time_steps = []
    for dt in time_steps:
        times = pd.date_range("2025-01-01", periods=int(96/dt), freq=pd.Timedelta(hours=dt))
        response = simulate_one_r_one_c(pd.Series(5.,index=times), pd.Series(5.,index=times), 18.,5.4,4.,dt)
        hours = np.arange(len(response))*dt
        exact_at_times = exact_constant_input(hours,18.,5.,5.,5.4,4.)
        error = None  # Replace with Euler minus exact at these times.
        euler_results[dt] = pd.DataFrame({"state_time":response.index,"elapsed_hours":hours,
            "euler_c":response.to_numpy(),"exact_c":exact_at_times,"error_c":error})
        ratio = dt/(5.4*4)
        euler_rows.append({"dt_hours":dt,"a":1-ratio,"dt_over_tau":ratio,
                           "max_abs_error_c":np.max(np.abs(error)),"N_intervals":len(times),"N_states":len(response)})
    ''', 0))
    code('w61-euler-plot', '''
    euler_table = pd.DataFrame(euler_rows)
    display(euler_table.round(6))
    export_table(euler_table, "euler_summary.csv", index=False)
    fig, axes = plt.subplots(2,1,figsize=(12,5.5),sharex=True)
    dense_hours = np.linspace(0,96,1000)
    axes[0].plot(dense_hours,exact_constant_input(dense_hours,18,5,5,5.4,4),color="black",label="Exact constant-input solution")
    all_euler = []
    for row in euler_rows:
        dt = row["dt_hours"]; values = euler_results[dt]
        label = f"dt={dt:g} h; a={row['a']:.3f}; dt/τ={row['dt_over_tau']:.3f}"
        line, = axes[0].plot(values.elapsed_hours, values.euler_c, ".-", label=label, ms=5)
        axes[1].plot(values.elapsed_hours, values.error_c.abs(), ".-", color=line.get_color(), label=f"dt={dt:g} h")
        all_euler.append(values.assign(dt_hours=dt))
    axes[0].set(ylabel="Temperature (°C)",title="Same physical model; errors at each method's actual sample times")
    axes[0].legend(fontsize=9,ncol=2,loc="lower right")
    axes[1].set(xlabel="Elapsed time (h)",ylabel="Absolute error (K)",xlim=(0,96))
    axes[1].legend(fontsize=10,ncol=3)
    export_table(pd.concat(all_euler,ignore_index=True), "euler_states.csv", index=False)
    save_figure(fig,"F61-08",period="Idealised 96 h experiment",units="°C;K;h",case="Euler vs exact",action="Task E step list and errors")
    ''')
    md('w61-euler-interpret', r'''
    **Interpret:** $\tau=21.6$ h. The 24 h step has negative $a$ but $|a|<1$:
    it oscillates about equilibrium and is asymptotically stable. Its accuracy is
    a separate question. Use the computed maximum errors to compare resolution.
    A finer step cannot repair omitted gains or an incorrect physical model.
    ''')
    md('w61-m6', r'''
    ## M6. The supplied two-state reference plant (slides 30–32)

    `TwoStateReferencePlant` is a separate evaluation model, inspired by the
    BOPTEST dwelling but not an exact reduction or validated reproduction.
    Its air and fabric states exchange heat; outdoors exchanges with fabric and
    ventilates air. Solar/internal gains and heating are allocated to both states.
    A heat-pump map converts dimensionless modulation to electricity, COP and heat.

    For the core comparison, disable both noise sources and initialise air and
    fabric at 20°C. Gains remain in the plant and are omitted in the compact
    model. Fabric temperature is a **hidden teaching diagnostic**, not a later
    controller measurement. Realised useful heat is shared here for diagnosis;
    a live controller would not know future realised COP.
    ''')
    code('w61-f09', r'''
    fig, ax = diagram((12,5))
    box(ax,.1,2.6,1.3,1.0,"Outdoor\n$T_o$")
    box(ax,3.5,2.5,2.0,1.2,"Fabric $T_f$\nHidden state")
    box(ax,7.4,2.5,1.8,1.2,"Air $T_a$\nMeasured state")
    ax.plot([1.5,3.4],[3.1,3.1],color="black"); ax.text(2.45,3.55,"$R_{of}=6$ K/kW",ha="center",fontsize=12)
    ax.plot([5.6,7.3],[3.1,3.1],color="black"); ax.text(6.45,3.55,"$R_{af}=0.8$ K/kW",ha="center",fontsize=12)
    ax.plot([.75,.75,8.3,8.3],[3.7,5,5,3.8],color="0.4",linestyle="--")
    ax.text(5,5.25,"Ventilation: $G_v=0.08$ kW/K; outdoors ↔ air",ha="center",fontsize=12)
    ax.text(4.0,2.05,"$C_f=18$ kWh/K",ha="right",fontsize=12)
    ax.text(7.8,2.05,"$C_a=0.8$ kWh/K",ha="right",fontsize=12)
    arrow(ax,(4.5,.9),(4.5,2.35),"",COLOURS["heat"])
    arrow(ax,(8.3,.9),(8.3,2.35),"",COLOURS["heat"])
    ax.text(4.4,.5,"0.75 $q_{th}$ + 0.55 $q_{gain}$",ha="center",fontsize=12)
    ax.text(8.0,.5,"0.25 $q_{th}$ + 0.45 $q_{gain}$",ha="center",fontsize=12)
    ax.text(2,1.2,"$q_{gain}$ = solar\n+ internal",ha="center",fontsize=12)
    arrow(ax,(9.25,3.1),(9.9,3.1))
    ax.text(9.55,4.1,"Air sensor\n(noise off)",ha="center",fontsize=11)
    save_figure(fig,"F61-09",["course_utils/reference_building.py","data/reference_building/reference_plant_parameters.json"],
                units="K/kW;kWh/K;kW/K;kW;°C",case="Supplied two-state plant; noiseless core")
    ''')
    code('w61-plant-run', '''
    parameters = ReferenceBuildingParameters(**json.loads((ROOT/"data/reference_building/reference_plant_parameters.json").read_text()))
    plant = TwoStateReferencePlant(parameters=parameters, step_hours=.5,
        measurement_noise_std_c=0.0, process_noise_std_c=0.0, random_state=12)
    measurement = plant.reset(20.0,20.0)
    state_records = [{"state_time":boundary_index[0],"air_true_c":20.0,"fabric_c":20.0,"air_measured_c":measurement}]
    input_records = []
    for timestamp, row in conditions.iterrows():
        modulation = .45 if (timestamp.hour < 8 or timestamp.hour >= 17) else 0.0
        result = plant.step(modulation,row.outdoor_temperature_c,row.solar_irradiance_w_m2,row.internal_gains_kw)
        interval_end = timestamp + pd.Timedelta(hours=step_hours)
        state_records.append({"state_time":interval_end,"air_true_c":result["indoor_temperature_true_c"],
                              "fabric_c":result["fabric_temperature_c"],"air_measured_c":result["indoor_temperature_measured_c"]})
        input_records.append({"interval_start":timestamp,"interval_end":interval_end,"dt_hours":step_hours,
            "modulation":modulation,"P_el_kw":result["heat_pump_electric_kw"],"q_th_kw":result["delivered_heat_kw"],
            "COP":result["cop"],"T_out_c":row.outdoor_temperature_c,
            "solar_irradiance_w_m2":row.solar_irradiance_w_m2,"solar_gain_kw":result["solar_gain_kw"],
            "internal_gains_kw":row.internal_gains_kw})
    reference_states = pd.DataFrame(state_records).set_index("state_time")
    reference_inputs = pd.DataFrame(input_records).set_index("interval_start")
    display(reference_inputs.head())
    export_table(reference_inputs,"reference_inputs.csv")
    export_table(reference_states,"reference_states.csv")
    ''')
    md('w61-task-f-text', '''
    ### Task F — compare models under common inputs (slide 31; 12 min)

    Feed `reference_inputs.T_out_c` and `reference_inputs.q_th_kw` into your
    compact simulator, with initial air temperature 20°C, R=5.4, C=4 and dt=0.5 h.
    Keep all 97 states. Interpret differences using the mechanisms each model
    represents; this is a structural/omitted-input comparison, not measurement noise.
    ''')
    task('F', '''
    temperature_compact = simulate_one_r_one_c(reference_inputs.T_out_c, reference_inputs.q_th_kw,
                            20.0, controller_resistance, controller_capacitance, step_hours)
    ''', '''
    # TODO F: simulate the compact model using the same realised useful heat and outdoor inputs.
    raise NotImplementedError("Task F: use reference_inputs, common initial air state and all intervals.")
    ''')
    code('w61-comparison', '''
    comparison_states = reference_states.copy()
    comparison_states["compact_c"] = temperature_compact
    model_gap = comparison_states.air_true_c-comparison_states.compact_c
    comparison_summary = pd.DataFrame([{"N_intervals":len(reference_inputs),"N_states":len(comparison_states),
        "air_gap_rmse_K_excluding_initial":np.sqrt(np.mean(model_gap.iloc[1:]**2)),
        "air_gap_max_abs_K":model_gap.abs().max(),
        "electric_energy_kWh":(reference_inputs.P_el_kw*reference_inputs.dt_hours).sum(),
        "useful_heat_energy_kWh":(reference_inputs.q_th_kw*reference_inputs.dt_hours).sum()}])
    display(comparison_summary.round(4))
    export_table(comparison_states,"comparison_states.csv")
    export_table(comparison_summary,"comparison_summary.csv",index=False)
    fig,axes = plt.subplots(2,1,figsize=(12,5.5),sharex=True,height_ratios=[2,1])
    axes[0].plot(reference_states.index,reference_states.air_true_c,color="black",label="Reference air (true; noise off)")
    axes[0].plot(temperature_compact.index,temperature_compact,color=COLOURS["compact"],label="Compact air (gains omitted)")
    axes[0].plot(reference_states.index,reference_states.fabric_c,"--",color=COLOURS["fabric"],label="Hidden fabric: teaching diagnostic")
    axes[0].set(ylabel="Temperature (°C)",title="Matched initial air state and inputs; 97 boundary states")
    axes[1].stairs(reference_inputs.q_th_kw,boundary_index,color=COLOURS["heat"],label="Delivered useful heat")
    axes[1].stairs(reference_inputs.P_el_kw,boundary_index,color=COLOURS["electric"],label="Heat-pump electricity")
    axes[1].set(ylabel="Power (kW)",xlabel="Boundary / interval time")
    time_axes(axes)
    save_figure(fig,"F61-10",["data/reference_building/operation_conditions.csv","course_utils/reference_building.py",
                "data/reference_building/reference_plant_parameters.json"],str(boundary_index[0])+" to "+str(boundary_index[-1]),
                "°C;kW;h", "Noiseless two-state plant vs 1R1C; same realised inputs", "Task F common-input call")
    ''')
    md('w61-m7', r'''
    ## M7. Model contract and handover (slides 33–34)

    In three sentences, cite one result, explain its mechanism and state one
    limitation. Distinguish approximation error from omitted mechanisms. The
    comparison with another synthetic model does not establish real-building accuracy.

    Before using the model, document: equation; kW/h/K units; dt; N-to-N+1 timing;
    state and initial condition; useful heat versus electricity; disturbances;
    parameter source; omitted gains; and intended operating range.

    In 6.2 we estimate effective parameters and test later observations. No
    parameter fitting or controller-model export is required today. Week 7 uses
    an instructor-supplied common model so fit differences do not confound control
    exercises. Its existing JSON was fitted on all 28 synthetic days, not an
    independent held-out validation result. Keep that shared file unchanged.
    ''')
    code('w61-manifest', '''
    if EXPORT_ARTEFACTS:
        pd.DataFrame(figure_manifest).sort_values("id").to_csv(ROOT/"teaching/week6_day1_figure_manifest.csv",index=False)
    ''')
    md('w61-optional', r'''
    ## Optional self-study (slides 36–39; outside the 200 minutes)

    **Exact discretisation.** For constant interval inputs,
    $a_e=\exp(-\Delta t/(RC))$, $b_e=1-a_e$, $c_e=R(1-a_e)$.
    These coefficients differ from Euler's. Check how recovering R,C would change.

    **Two-state balances.** Use the supplied plant diagram and source. Its air
    balance includes $(T_f-T_a)/R_{af}+G_v(T_o-T_a)+0.45q_{gain}+0.25q_{th}$;
    the fabric balance includes $(T_o-T_f)/R_{of}+(T_a-T_f)/R_{af}+0.55q_{gain}+0.75q_{th}$.
    Divide each net heat by the corresponding capacitance to obtain its derivative.

    **Detailed simulators.** OCHRE and BOPTEST provide richer building/equipment
    contexts. No installation or online service is required here. The supplied
    plant is a teaching surrogate, not validated ground truth.

    **Unit conversion.** $R_{K/kW}=1000R_{K/W}$;
    $C_{kWh/K}=C_{J/K}/3{,}600{,}000$; $q_W=1000q_{kW}$;
    $\Delta t_s=3600\Delta t_h$. Verify an independent transition in either system.

    The next supplied examples keep unstable Euler dynamics and sensor noise
    separate from the core figures. For noise, create fresh plant instances;
    `reset()` does not rewind the generator.
    ''')
    code('w61-optional-instability', '''
    optional_index = pd.date_range("2025-01-01",periods=2,freq="48h")
    unstable = simulate_one_r_one_c(pd.Series(5.,index=optional_index),pd.Series(5.,index=optional_index),18.,5.4,4.,48.)
    fig,ax=plt.subplots(figsize=(10,4))
    ax.plot([0,48,96],unstable,"o-",color="tab:red",label="Euler dt=48 h; a=-1.222; unstable")
    ax.axhline(32,color="black",linestyle="--",label="Equilibrium")
    ax.set(xlabel="Elapsed time (h)",ylabel="Temperature (°C)",title="Optional: unstable case on its own scale")
    ax.legend(); fig.tight_layout()
    if EXPORT_ARTEFACTS:
        for ext in ["png","svg"]:fig.savefig(FIGURE_DIR/f"optional_euler_48h.{ext}",dpi=220,bbox_inches="tight")
    plt.show(); plt.close(fig)
    ''')
    code('w61-optional-noise', '''
    # A fresh instance per demonstration makes repeated runs deterministic.
    noisy_plant=TwoStateReferencePlant(parameters=parameters,step_hours=.5,
        measurement_noise_std_c=.05,process_noise_std_c=0.,random_state=12)
    noisy_measurements=[noisy_plant.reset(20.,20.)]
    for timestamp,row in conditions.iterrows():
        modulation=reference_inputs.loc[timestamp,"modulation"]
        result=noisy_plant.step(modulation,row.outdoor_temperature_c,row.solar_irradiance_w_m2,row.internal_gains_kw)
        noisy_measurements.append(result["indoor_temperature_measured_c"])
    optional_noise = pd.DataFrame({"true_air_c":reference_states.air_true_c,"measured_air_c":noisy_measurements})
    export_table(optional_noise,"optional_measurement_noise.csv")
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(optional_noise.index,optional_noise.true_air_c,color="black",label="True air")
    ax.plot(optional_noise.index,optional_noise.measured_air_c,".",ms=3,color="tab:blue",label="Fresh sensor-noise example; σ=0.05°C")
    ax.set(ylabel="Temperature (°C)",xlabel="State time",title="Optional: measurement noise only; process noise disabled")
    time_axes(ax); fig.tight_layout()
    if EXPORT_ARTEFACTS:
        for ext in ["png","svg"]:fig.savefig(FIGURE_DIR/f"optional_measurement_noise.{ext}",dpi=220,bbox_inches="tight")
    plt.show(); plt.close(fig)
    ''')
    md('w61-references', '''
    ### References and source credits (slide 35)

    Reference dataset and plant assumptions: [local documentation](data/reference_building/README.md)
    and `course_utils/reference_building.py`. Newly drawn diagrams describe the
    teaching model; no source dataset is implied for instructional diagrams.

    Retain the lecture's references: Bacher and Madsen (2011), *Energy and
    Buildings* 43, 1511–1522; Drgoňa et al. (2020), *Annual Reviews in Control*
    50, 190–232; Blonsky et al. (2021), *Applied Energy* 290, 116732 (optional OCHRE).
    ''')
    return nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})


if __name__ == '__main__':
    for solution in [False, True]:
        path = ROOT / (STEM + ('_solution' if solution else '') + '.ipynb')
        nbf.write(make_notebook(solution), path)
    with (ROOT / 'teaching/week6_day1_task_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task_id', 'section', 'slide', 'minutes_including_debrief', 'student_completion', 'cell_id'])
        for letter, section, slide, minutes, action in TASKS:
            writer.writerow([letter, section, slide, minutes, action, 'w61-task-' + letter.lower()])
    print('Built Tutorial 6.1 pair only.')
