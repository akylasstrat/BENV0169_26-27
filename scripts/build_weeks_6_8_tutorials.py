"""Build the paired student and solution notebooks for Tutorials 6.1--8.2."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def choose(solution, completed, student):
    return completed if solution else student


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def setup_cell(extra_imports=""):
    return code(
        f"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
{extra_imports}

ROOT = Path.cwd()
if not (ROOT / "data").exists():
    raise FileNotFoundError("Run this notebook from the repository root.")

plt.rcParams.update({{"figure.figsize": (10, 4), "figure.dpi": 110}})
"""
    )


def tutorial_6_1(solution):
    step_function = choose(
        solution,
        """
def one_r_one_c_step(
    indoor_temperature_c,
    outdoor_temperature_c,
    delivered_heat_kw,
    resistance_k_per_kw,
    capacitance_kwh_per_k,
    step_hours,
):
    heat_loss_kw = (outdoor_temperature_c - indoor_temperature_c) / resistance_k_per_kw
    temperature_change_c = step_hours * (heat_loss_kw + delivered_heat_kw) / capacitance_kwh_per_k
    return indoor_temperature_c + temperature_change_c
""",
        """
def one_r_one_c_step(
    indoor_temperature_c,
    outdoor_temperature_c,
    delivered_heat_kw,
    resistance_k_per_kw,
    capacitance_kwh_per_k,
    step_hours,
):
    \"\"\"Return the indoor temperature at the end of one interval.\"\"\"
    # TODO 1.1: calculate heat exchange with outdoors.
    # TODO 1.2: calculate the temperature change and return the next temperature.
    raise NotImplementedError("Complete the $1R1C$ update.")
""",
    )
    simulation_function = choose(
        solution,
        """
def simulate_one_r_one_c(
    outdoor_temperature_c,
    delivered_heat_kw,
    initial_temperature_c,
    resistance_k_per_kw,
    capacitance_kwh_per_k,
    step_hours,
):
    temperatures = [float(initial_temperature_c)]
    for outdoor, heat in zip(outdoor_temperature_c.iloc[:-1], delivered_heat_kw.iloc[:-1]):
        temperatures.append(
            one_r_one_c_step(
                temperatures[-1], outdoor, heat,
                resistance_k_per_kw, capacitance_kwh_per_k, step_hours
            )
        )
    return pd.Series(temperatures, index=outdoor_temperature_c.index, name="indoor_temperature_c")
""",
        """
def simulate_one_r_one_c(
    outdoor_temperature_c,
    delivered_heat_kw,
    initial_temperature_c,
    resistance_k_per_kw,
    capacitance_kwh_per_k,
    step_hours,
):
    \"\"\"Roll the model forwards over aligned outdoor-temperature and heat series.\"\"\"
    temperatures = [float(initial_temperature_c)]
    # TODO 2.1: loop over the intervals and append each predicted temperature.
    # Hint: feed the previous model prediction into one_r_one_c_step().
    raise NotImplementedError("Complete the free-rollout simulation.")
""",
    )
    return notebook(
        [
            md(
                """
# Tutorial 6.1: Control-oriented building modelling and simulation

This tutorial turns a continuous thermal balance into a small discrete model that can be simulated repeatedly. The controller model is deliberately simpler than the supplied reference plant.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. implement a discrete $1R1C$ temperature update;
2. simulate a free temperature trajectory;
3. explain how $R$, $C$ and the sampling interval affect the response; and
4. distinguish a controller model from a reference plant.

## Indicative schedule

- set-up and energy balance: 15 minutes;
- discrete model and rollout: 25 minutes;
- step-response experiments: 25 minutes;
- comparison with the reference plant: 20 minutes;
- discussion: 5 minutes.
"""
            ),
            md("## 1. Set-up and operating conditions"),
            setup_cell("from course_utils.reference_building import TwoStateReferencePlant"),
            code(
                """
conditions = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_conditions.csv",
    index_col="timestamp",
    parse_dates=True,
).iloc[: 2 * 48]

display(conditions.head())
conditions[["outdoor_temperature_c", "comfort_min_c", "comfort_max_c"]].plot()
plt.ylabel("Temperature [°C]")
plt.title("Two days of operating conditions")
plt.show()
"""
            ),
            md(
                r"""
## 2. Task 1 — one discrete $1R1C$ update

For resistance $R$ in K/kW, capacitance $C$ in kWh/K and interval length $\Delta t$ in hours:

$$
T_{k+1}=T_k+\frac{\Delta t}{C}\left(\frac{T_{o,k}-T_k}{R}+q_{th,k}\right)
$$

Heating is positive. The input $q_{th,k}$ is useful heat delivered to the zone, not electrical power.
"""
            ),
            code(step_function),
            code(
                """
example_next_temperature = one_r_one_c_step(
    indoor_temperature_c=20.0,
    outdoor_temperature_c=5.0,
    delivered_heat_kw=4.0,
    resistance_k_per_kw=5.4,
    capacitance_kwh_per_k=4.0,
    step_hours=0.5,
)
print(f"Next temperature: {example_next_temperature:.3f} °C")
"""
            ),
            md(
                """
## 3. Task 2 — free rollout

A **free rollout** starts from one observed initial temperature and then feeds each model prediction into the next update. No later indoor-temperature measurements are used to reset the model. This is also called a recursive multi-step simulation.

Use your one-step update repeatedly over the two-day input series. Start from the same indoor temperature in both cases and compare (i) no heating with (ii) a prescribed morning and evening heating schedule under the same outdoor-temperature conditions. Each predicted indoor temperature becomes the initial condition for the next interval; the model is not reset from measurements.

The upper panel compares the two simulated indoor-temperature trajectories with outdoor temperature. The lower panel shows the useful-heat schedule used in the scheduled-heating case.
"""
            ),
            code(simulation_function),
            code(
                """
step_hours = 0.5
controller_resistance = 5.4
controller_capacitance = 4.0

heat_schedule = pd.Series(0.0, index=conditions.index)
heat_schedule.loc[(conditions.index.hour >= 5) & (conditions.index.hour < 8)] = 6.0
heat_schedule.loc[(conditions.index.hour >= 17) & (conditions.index.hour < 22)] = 4.0

temperature_no_heat = simulate_one_r_one_c(
    conditions["outdoor_temperature_c"],
    pd.Series(0.0, index=conditions.index),
    20.0,
    controller_resistance,
    controller_capacitance,
    step_hours,
)
temperature_scheduled = simulate_one_r_one_c(
    conditions["outdoor_temperature_c"],
    heat_schedule,
    20.0,
    controller_resistance,
    controller_capacitance,
    step_hours,
)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
axes[0].plot(temperature_no_heat, label="No heating", color = 'tab:blue')
axes[0].plot(temperature_scheduled, label="Scheduled heating", color = 'tab:orange')
axes[0].plot(conditions["outdoor_temperature_c"], label="Outdoor", alpha=0.7)
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].step(heat_schedule.index, heat_schedule, where="post", color = 'tab:orange')
axes[1].set_ylabel("Useful heat [kW]")
axes[1].set_xlabel("Time")
plt.show()
"""
            ),
            md(
                """
## 4. Task 3 — parameter experiments

This is a new, idealised sensitivity experiment rather than an exact repetition of Task 2. The outdoor temperature is held at 5 °C, useful heating is held at 5 kW, and every simulation starts at 18 °C. The left panel varies $R$ while holding $C$ fixed; the right panel varies $C$ while holding $R$ fixed. Each line is a free rollout of indoor temperature over 48 hours.

Changing one parameter at a time isolates its effect. Predict how the curves should differ before running the cell.
"""
            ),
            code(
                """
experiment_index = pd.date_range("2025-01-01", periods=96, freq="30min")
experiment_outdoor = pd.Series(5.0, index=experiment_index)
experiment_heat = pd.Series(5.0, index=experiment_index)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for resistance in [3.5, 5.4, 8.0]:
    response = simulate_one_r_one_c(
        experiment_outdoor, experiment_heat, 18.0,
        resistance, controller_capacitance, step_hours
    )
    axes[0].plot(response, label=f"R = {resistance} K/kW")
for capacitance in [2.0, 4.0, 8.0]:
    response = simulate_one_r_one_c(
        experiment_outdoor, experiment_heat, 18.0,
        controller_resistance, capacitance, step_hours
    )
    axes[1].plot(response, label=f"C = {capacitance} kWh/K")
for axis in axes:
    axis.set_xlabel("Time")
    axis.set_ylabel("Indoor temperature [°C]")
    axis.legend()
axes[0].set_title("Changing resistance")
axes[1].set_title("Changing capacitance")
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """
## 5. Supplied reference plant

The reference plant has separate indoor-air and fabric temperatures, variable COP, ventilation, solar gains, internal gains and reproducible noise. Its `reset` and `step` interface is kept separate from the controller model so that it can later be replaced by BOPTEST.
"""
            ),
            code(
                """
plant = TwoStateReferencePlant(
    step_hours=0.5,
    measurement_noise_std_c=0.05,
    process_noise_std_c=0.01,
    random_state=12,
)
measurement = plant.reset(20.0, 20.0)
records = []

for timestamp, row in conditions.iterrows():
    modulation = 0.45 if (timestamp.hour < 8 or timestamp.hour >= 17) else 0.0
    result = plant.step(
        modulation,
        row["outdoor_temperature_c"],
        row["solar_irradiance_w_m2"],
        row["internal_gains_kw"],
    )
    records.append({"timestamp": timestamp, **result, "modulation": modulation})

reference_results = pd.DataFrame(records).set_index("timestamp")
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
axes[0].plot(reference_results["indoor_temperature_true_c"], label="True indoor")
axes[0].plot(reference_results["indoor_temperature_measured_c"], ".", ms=2, label="Measured indoor")
axes[0].plot(reference_results["fabric_temperature_c"], label="Hidden fabric")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].step(reference_results.index, reference_results["heat_pump_electric_kw"], where="post")
axes[1].set_ylabel("Heat-pump electricity [kW]")
axes[1].set_xlabel("Time")
plt.show()
"""
            ),
            md(
                """
## Discussion

1. Which parameter mainly changes the steady temperature response?
2. Which parameter mainly changes response speed?
3. Why does the measured reference-plant temperature look less smooth?
4. Why should the optimisation model remain simpler than the reference plant?
"""
            ),
        ]
    )


def tutorial_6_2(solution):
    regression = choose(
        solution,
        """
def make_regression_table(data):
    table = data.copy()
    table["temperature_change_c"] = (
        table["indoor_temperature_c"].shift(-1) - table["indoor_temperature_c"]
    )
    table["temperature_difference_c"] = (
        table["outdoor_temperature_c"] - table["indoor_temperature_c"]
    )
    return table.iloc[:-1]
""",
        """
def make_regression_table(data):
    \"\"\"Create aligned predictors at interval k and target T_(k+1)-T_k.\"\"\"
    table = data.copy()
    # TODO 1.1: create temperature_change_c using the next indoor temperature.
    # TODO 1.2: create outdoor minus indoor temperature_difference_c.
    # TODO 1.3: remove the final row, whose next temperature is unavailable.
    raise NotImplementedError("Construct the aligned regression table.")
""",
    )
    bounded = choose(
        solution,
        """
def fit_bounded_rc(regression_table):
    design = regression_table[["temperature_difference_c", "delivered_heat_kw"]].to_numpy()
    target = regression_table["temperature_change_c"].to_numpy()
    result = lsq_linear(design, target, bounds=([1e-5, 1e-5], [0.4, 0.5]))
    if not result.success:
        raise RuntimeError(result.message)
    return result.x
""",
        """
def fit_bounded_rc(regression_table):
    \"\"\"Return positive loss and heating coefficients from bounded least squares.\"\"\"
    # TODO 2.1: build the two-column design matrix.
    # TODO 2.2: call lsq_linear with positive, physically meaningful bounds.
    # TODO 2.3: check result.success and return result.x.
    raise NotImplementedError("Fit the bounded RC coefficients.")
""",
    )
    recovery = choose(
        solution,
        """
def recover_rc_parameters(loss_coefficient, heat_coefficient, step_hours):
    capacitance_kwh_per_k = step_hours / heat_coefficient
    resistance_k_per_kw = heat_coefficient / loss_coefficient
    time_constant_hours = resistance_k_per_kw * capacitance_kwh_per_k
    return resistance_k_per_kw, capacitance_kwh_per_k, time_constant_hours
""",
        """
def recover_rc_parameters(loss_coefficient, heat_coefficient, step_hours):
    \"\"\"Recover effective R, C and tau from the fitted difference equation.\"\"\"
    # TODO 3: implement the three expressions given in the Markdown above.
    raise NotImplementedError("Recover the effective physical parameters.")
""",
    )
    rollout = choose(
        solution,
        """
def one_step_predictions(data, loss_coefficient, heat_coefficient):
    current = data["indoor_temperature_c"].iloc[:-1].to_numpy()
    outdoor = data["outdoor_temperature_c"].iloc[:-1].to_numpy()
    heat = data["delivered_heat_kw"].iloc[:-1].to_numpy()
    values = current + loss_coefficient * (outdoor - current) + heat_coefficient * heat
    return pd.Series(values, index=data.index[1:], name="one_step_prediction_c")


def rollout_predictions(data, loss_coefficient, heat_coefficient):
    predictions = [float(data["indoor_temperature_c"].iloc[0])]
    for position in range(len(data) - 1):
        predicted_next = (
            predictions[-1]
            + loss_coefficient * (data["outdoor_temperature_c"].iloc[position] - predictions[-1])
            + heat_coefficient * data["delivered_heat_kw"].iloc[position]
        )
        predictions.append(predicted_next)
    return pd.Series(predictions, index=data.index, name="rollout_prediction_c")
""",
        """
def one_step_predictions(data, loss_coefficient, heat_coefficient):
    \"\"\"Reset from each measured temperature and predict the next value.\"\"\"
    # TODO 4.1: implement the vectorised one-step prediction.
    raise NotImplementedError("Create one-step predictions.")


def rollout_predictions(data, loss_coefficient, heat_coefficient):
    \"\"\"Initialise once and feed each prediction into the next step.\"\"\"
    # TODO 4.2: implement the recursive rollout.
    raise NotImplementedError("Create rollout predictions.")
""",
    )
    measurement_prep = choose(
        solution,
        """
def prepare_house_measurements(raw_data):
    columns = [
        "T_Average (degC)",
        "T_External (degC)",
        "P_tot (W)",
        "Solar (W/m2)",
    ]
    data = raw_data[columns].copy()
    data = data.interpolate(method="time", limit_direction="both")
    if data.isna().any().any():
        raise ValueError("The selected measurement columns still contain missing values.")

    table = pd.DataFrame(index=data.index)
    table["indoor_temperature_c"] = data["T_Average (degC)"]
    table["outdoor_temperature_c"] = data["T_External (degC)"]
    table["power_proxy_kw"] = data["P_tot (W)"] / 1000
    table["solar_proxy_kw_m2"] = data["Solar (W/m2)"] / 1000
    table["next_indoor_temperature_c"] = table["indoor_temperature_c"].shift(-1)
    table["temperature_change_c"] = (
        table["next_indoor_temperature_c"] - table["indoor_temperature_c"]
    )
    table["temperature_difference_c"] = (
        table["outdoor_temperature_c"] - table["indoor_temperature_c"]
    )
    return table.iloc[:-1]
""",
        """
def prepare_house_measurements(raw_data):
    \"\"\"Prepare aligned indoor, outdoor, power and solar measurement columns.\"\"\"
    # TODO E1.1: select the four columns named in the Markdown above.
    # TODO E1.2: interpolate the short gaps separately within this measurement period.
    # TODO E1.3: convert power and solar irradiance from W to kW-based units.
    # TODO E1.4: create the next temperature, temperature change and temperature difference.
    # TODO E1.5: remove the final row, whose next temperature is unavailable.
    raise NotImplementedError("Prepare the occupied-house measurement table.")
""",
    )
    measurement_fit = choose(
        solution,
        """
def fit_house_measurement_model(measurement_table):
    feature_columns = [
        "temperature_difference_c",
        "power_proxy_kw",
        "solar_proxy_kw_m2",
    ]
    design = measurement_table[feature_columns].to_numpy()
    target = measurement_table["temperature_change_c"].to_numpy()
    result = lsq_linear(
        design,
        target,
        bounds=([1e-6, 1e-6, 0.0], [0.2, 0.5, 2.0]),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x
""",
        """
def fit_house_measurement_model(measurement_table):
    \"\"\"Fit positive temperature-loss, power-proxy and solar coefficients.\"\"\"
    # TODO E2.1: construct the three-column design matrix and target vector.
    # TODO E2.2: fit bounded coefficients with lsq_linear.
    # TODO E2.3: check result.success and return result.x.
    raise NotImplementedError("Fit the occupied-house measurement model.")
""",
    )
    measurement_predict = choose(
        solution,
        """
def predict_house_measurements(measurement_table, coefficients):
    loss_coefficient, power_coefficient, solar_coefficient = coefficients
    valid_index = measurement_table.index + pd.Timedelta(minutes=10)

    one_step_values = (
        measurement_table["indoor_temperature_c"].to_numpy()
        + measurement_table[
            ["temperature_difference_c", "power_proxy_kw", "solar_proxy_kw_m2"]
        ].to_numpy()
        @ coefficients
    )
    one_step = pd.Series(one_step_values, index=valid_index, name="one_step_c")

    predicted_temperature = float(measurement_table["indoor_temperature_c"].iloc[0])
    rollout_values = []
    for _, row in measurement_table.iterrows():
        predicted_temperature = (
            predicted_temperature
            + loss_coefficient
            * (row["outdoor_temperature_c"] - predicted_temperature)
            + power_coefficient * row["power_proxy_kw"]
            + solar_coefficient * row["solar_proxy_kw_m2"]
        )
        rollout_values.append(predicted_temperature)

    rollout = pd.Series(rollout_values, index=valid_index, name="free_rollout_c")
    actual = pd.Series(
        measurement_table["next_indoor_temperature_c"].to_numpy(),
        index=valid_index,
        name="measured_c",
    )
    return actual, one_step, rollout
""",
        """
def predict_house_measurements(measurement_table, coefficients):
    \"\"\"Return measured, one-step and free-rollout temperature series.\"\"\"
    # TODO E3.1: calculate one-step predictions using measured T_k each time.
    # TODO E3.2: calculate a free rollout using only the first measured temperature.
    # TODO E3.3: return aligned actual, one-step and rollout Series.
    raise NotImplementedError("Evaluate the occupied-house measurement model.")
""",
    )
    return notebook(
        [
            md(
                """
# Tutorial 6.2: RC model identification and validation

This tutorial estimates a compact $1R1C$ controller model from noisy measurements generated by the independent reference plant. The model is evaluated on a later, untouched period.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. align thermal inputs with the temperature transition they cause;
2. fit bounded coefficients using physical knowledge;
3. recover effective $R$, $C$ and $\tau$ values; and
4. distinguish one-step validation from free rollout.

## Indicative schedule

- data inspection and alignment: 15 minutes;
- least-squares fitting: 25 minutes;
- parameter recovery: 15 minutes;
- validation and residuals: 30 minutes;
- discussion: 5 minutes.

The optional occupied-house extension adds approximately 30 minutes and can also be completed independently after class.
"""
            ),
            md("## 1. Set-up and data"),
            setup_cell("from scipy.optimize import lsq_linear"),
            code(
                """
data = pd.read_csv(
    ROOT / "data" / "reference_building" / "identification_data.csv",
    index_col="timestamp",
    parse_dates=True,
).sort_index()

print(f"Rows: {len(data):,}")
print(f"Period: {data.index.min()} to {data.index.max()}")
print(f"Duplicate timestamps: {data.index.duplicated().sum()}")
print(f"Missing values: {int(data.isna().sum().sum())}")
display(data.head())

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
window = data.iloc[: 7 * 48]
axes[0].plot(window["indoor_temperature_c"], label="Indoor")
axes[0].plot(window["outdoor_temperature_c"], label="Outdoor")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].step(window.index, window["delivered_heat_kw"], where="post")
axes[1].set_ylabel("Useful heat [kW]")
axes[2].plot(window["solar_irradiance_w_m2"])
axes[2].set_ylabel("Irradiance [W/m²]")
axes[2].set_xlabel("Time")
plt.show()
"""
            ),
            md(
                r"""
## 2. Task 1 — aligned regression data

We fit the difference form

$$
T_{k+1}-T_k=b(T_{o,k}-T_k)+cq_{th,k}+\varepsilon_k
$$

where $b=\Delta t/(RC)$ and $c=\Delta t/C$. Inputs on row $k$ must be paired with the transition from $T_k$ to $T_{k+1}$.
"""
            ),
            code(regression),
            code(
                """
regression_table = make_regression_table(data)
split_time = data.index.min() + pd.Timedelta(days=21)
calibration = regression_table.loc[regression_table.index < split_time]
validation_data = data.loc[data.index >= split_time]

print(f"Calibration rows: {len(calibration):,}")
print(f"Validation rows: {len(validation_data):,}")
"""
            ),
            md("## 3. Unconstrained worked example"),
            code(
                """
design_calibration = calibration[["temperature_difference_c", "delivered_heat_kw"]].to_numpy()
target_calibration = calibration["temperature_change_c"].to_numpy()
unconstrained_coefficients, *_ = np.linalg.lstsq(
    design_calibration, target_calibration, rcond=None
)
print("Unconstrained coefficients:", unconstrained_coefficients)
"""
            ),
            md(
                """
## 4. Task 2 — bounded estimation

Positive bounds encode the expected direction of outdoor heat exchange and heating. Bounds do not guarantee that the model is adequate; they prevent clearly impossible coefficient signs.
"""
            ),
            code(bounded),
            code(
                """
loss_coefficient, heat_coefficient = fit_bounded_rc(calibration)
print(f"Loss coefficient b: {loss_coefficient:.5f}")
print(f"Heating coefficient c: {heat_coefficient:.5f} °C/kW per step")
"""
            ),
            md(
                r"""
## 5. Task 3 — recover effective parameters

$$
C=\frac{\Delta t}{c},\qquad R=\frac{c}{b},\qquad \tau=RC
$$

These are effective parameters for the selected model, sampling interval and operating data. They are not individual material properties.
"""
            ),
            code(recovery),
            code(
                """
estimated_r, estimated_c, estimated_tau = recover_rc_parameters(
    loss_coefficient, heat_coefficient, step_hours=0.5
)
print(f"R = {estimated_r:.3f} K/kW")
print(f"C = {estimated_c:.3f} kWh/K")
print(f"tau = {estimated_tau:.1f} h")
"""
            ),
            md(
                """
## 6. Task 4 — one-step and rollout validation

A one-step prediction uses the measured $T_k$ as the starting point for every transition. A **free rollout** starts from one measured temperature and then uses each predicted temperature as the starting point for the next transition. It receives no later indoor-temperature measurements, so small model errors can accumulate over the validation period.
"""
            ),
            code(rollout),
            code(
                """
def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted values must have the same shape.")
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


one_step = one_step_predictions(validation_data, loss_coefficient, heat_coefficient)
rollout = rollout_predictions(validation_data, loss_coefficient, heat_coefficient)
actual_one_step = validation_data["indoor_temperature_c"].iloc[1:]
actual_rollout = validation_data["indoor_temperature_c"]

metrics = pd.DataFrame(
    {
        "RMSE [°C]": [
            rmse(actual_one_step, one_step),
            rmse(actual_rollout, rollout),
        ],
        "Bias [°C]": [
            float((one_step - actual_one_step).mean()),
            float((rollout - actual_rollout).mean()),
        ],
    },
    index=["One-step", "Free rollout"],
)
display(metrics.round(3))

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
axes[0].plot(actual_rollout, label="Measured", color="black")
axes[0].plot(one_step, label="One-step", alpha=0.8)
axes[0].plot(rollout, label="Free rollout", alpha=0.8)
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
rollout_residual = actual_rollout - rollout
axes[1].plot(rollout_residual)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Measured − rollout [°C]")
axes[1].set_xlabel("Time")
plt.show()
"""
            ),
            md("## 7. Residual diagnosis and supplied controller model"),
            code(
                """
residual_frame = pd.DataFrame(
    {
        "residual_c": rollout_residual,
        "heating_kw": validation_data["delivered_heat_kw"],
        "hour": validation_data.index.hour + validation_data.index.minute / 60,
    }
).dropna()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(residual_frame["hour"], residual_frame["residual_c"], s=8, alpha=0.5)
axes[0].set_xlabel("Hour of day")
axes[0].set_ylabel("Rollout residual [°C]")
axes[1].scatter(residual_frame["heating_kw"], residual_frame["residual_c"], s=8, alpha=0.5)
axes[1].set_xlabel("Delivered heat [kW]")
axes[1].set_ylabel("Rollout residual [°C]")
plt.tight_layout()
plt.show()

verified_model = json.loads(
    (ROOT / "data" / "reference_building" / "verified_controller_model.json").read_text()
)
display(pd.Series(verified_model, name="Verified model"))
"""
            ),
            md(
                r"""
## 8. Optional extension — occupied-house measurements

This extension applies the same identification workflow to measurements from an occupied semi-detached house in the UK. The two files contain five days at 10-minute resolution:

- November 2016, used for calibration;
- June 2017, used for independent validation without refitting.

The available `P_tot (W)` signal is a proxy for heat entering the building, not a direct measurement of useful space-heating output. The solar coefficient also absorbs glazing, shading and other effects. Treat the resulting coefficients and recovered $R$ and $C$ as effective parameters for this data and model.

Source: Hollick, F. and Wingfield, J. (2018), *Two periods of in-situ measurements from an occupied, semi-detached house in the UK*. https://doi.org/10.14324/000.ds.10087216
"""
            ),
            code(
                """
measurement_directory = ROOT / "data" / "tutorial_6_2"


def load_house_period(filename):
    period = pd.read_csv(measurement_directory / filename, index_col=0)
    period.index = pd.to_datetime(period.index, format="%d/%m/%Y %H:%M")
    return period.sort_index()


winter_raw = load_house_period("10mins_solpap_2016.csv")
summer_raw = load_house_period("10mins_solpap_2017.csv")

measurement_summary = pd.DataFrame(
    {
        "Rows": [len(winter_raw), len(summer_raw)],
        "Start": [winter_raw.index.min(), summer_raw.index.min()],
        "End": [winter_raw.index.max(), summer_raw.index.max()],
        "Missing selected values": [
            int(winter_raw[["T_Average (degC)", "T_External (degC)", "P_tot (W)", "Solar (W/m2)"]].isna().sum().sum()),
            int(summer_raw[["T_Average (degC)", "T_External (degC)", "P_tot (W)", "Solar (W/m2)"]].isna().sum().sum()),
        ],
    },
    index=["November 2016", "June 2017"],
)
display(measurement_summary)

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
axes[0].plot(winter_raw["T_Average (degC)"], label="Indoor average")
axes[0].plot(winter_raw["T_External (degC)"], label="Outdoor")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].plot(winter_raw["P_tot (W)"] / 1000)
axes[1].set_ylabel("Power proxy [kW]")
axes[2].plot(winter_raw["Solar (W/m2)"])
axes[2].set_ylabel("Solar [W/m²]")
axes[2].set_xlabel("Time")
plt.show()
"""
            ),
            md(
                r"""
### Extension Task E1 — prepare the measured data

Interpolate the short measurement gaps separately within each five-day period. This is retrospective model identification, so interpolation across an internal gap is available. Then form

$$
T_{k+1}-T_k=b(T_{o,k}-T_k)+cP^{proxy}_k+gS_k+\varepsilon_k.
$$

Use `P_tot (W)` in kW and scale `Solar (W/m2)` to kW/m² so the fitted coefficients remain numerically convenient.
"""
            ),
            code(measurement_prep),
            code(
                """
winter_measurements = prepare_house_measurements(winter_raw)
summer_measurements = prepare_house_measurements(summer_raw)
display(winter_measurements.head(3))
"""
            ),
            md(
                """
### Extension Task E2 — fit on November 2016

Fit the three coefficients using the winter period only. Do not use June 2017 to choose coefficients or bounds.
"""
            ),
            code(measurement_fit),
            code(
                """
house_coefficients = fit_house_measurement_model(winter_measurements)
house_loss_coefficient, house_power_coefficient, house_solar_coefficient = house_coefficients

house_step_hours = 1 / 6
house_effective_r = house_power_coefficient / house_loss_coefficient
house_effective_c = house_step_hours / house_power_coefficient
house_effective_tau = house_step_hours / house_loss_coefficient

print(f"Loss coefficient: {house_loss_coefficient:.6f}")
print(f"Power-proxy coefficient: {house_power_coefficient:.6f}")
print(f"Solar coefficient: {house_solar_coefficient:.6f}")
print(f"Effective R under the proxy assumption: {house_effective_r:.2f} K/kW")
print(f"Effective C under the proxy assumption: {house_effective_c:.2f} kWh/K")
print(f"Effective time constant: {house_effective_tau:.1f} h")
"""
            ),
            md(
                """
### Extension Task E3 — test without refitting

Compare one-step prediction and a free rollout on both periods. The 2017 period is a demanding test because it has warmer weather and much less gas use than the calibration period.
"""
            ),
            code(measurement_predict),
            code(
                """
house_results = {}
house_metrics = {}
for period_name, period_table in {
    "2016 calibration": winter_measurements,
    "2017 validation": summer_measurements,
}.items():
    actual, one_step_house, rollout_house = predict_house_measurements(
        period_table, house_coefficients
    )
    house_results[period_name] = (actual, one_step_house, rollout_house)
    house_metrics[period_name] = {
        "One-step RMSE [°C]": rmse(actual, one_step_house),
        "Free-rollout RMSE [°C]": rmse(actual, rollout_house),
        "Free-rollout bias [°C]": float((rollout_house - actual).mean()),
    }

house_metric_table = pd.DataFrame(house_metrics).T
display(house_metric_table.round(3))

actual, one_step_house, rollout_house = house_results["2017 validation"]
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
axes[0].plot(actual, label="Measured", color="black")
axes[0].plot(one_step_house, label="One-step", alpha=0.8)
axes[0].plot(rollout_house, label="Free rollout", alpha=0.8)
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].plot(actual - one_step_house, label="One-step residual")
axes[1].plot(actual - rollout_house, label="Free-rollout residual", alpha=0.8)
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_ylabel("Measured − predicted [°C]")
axes[1].set_xlabel("Time")
axes[1].legend()
plt.show()
"""
            ),
            md(
                """
### Extension questions

1. Why can the one-step RMSE remain small while the free rollout drifts substantially?
2. How do the June 2017 operating conditions differ from the November 2016 calibration period?
3. Why should coefficients based on `P_tot (W)` be treated as effective rather than physical parameters?
4. Which additional measurements would make the identification problem more physically interpretable?
"""
            ),
            md(
                """
## Discussion

1. Why is the rollout error larger than the one-step error?
2. Which omitted effects could explain a residual pattern by hour of day?
3. Why will later tutorials use the common verified model rather than every student's fitted model?
4. Over what horizon and operating range does the fitted model appear credible?
"""
            ),
        ]
    )


def tutorial_7_1(solution):
    optimiser = choose(
        solution,
        """
def optimise_heating(
    conditions,
    model,
    initial_temperature_c=20.0,
    soft_constraints=True,
    slack_penalty=25.0,
):
    n_steps = len(conditions)
    dt = model["sampling_interval_hours"]
    electric_power = cp.Variable(n_steps, nonneg=True)
    temperature = cp.Variable(n_steps + 1)
    lower_slack = cp.Variable(n_steps, nonneg=True)

    constraints = [
        temperature[0] == initial_temperature_c,
        electric_power <= model["heat_pump_max_electric_kw"],
    ]
    if not soft_constraints:
        constraints.append(lower_slack == 0)

    for step in range(n_steps):
        delivered_heat = model["nominal_cop"] * electric_power[step]
        constraints.append(
            temperature[step + 1]
            == temperature[step]
            + model["loss_coefficient_per_step"]
            * (conditions["outdoor_temperature_c"].iloc[step] - temperature[step])
            + model["heat_coefficient_c_per_kw_step"] * delivered_heat
        )
        if soft_constraints:
            constraints.append(
                temperature[step + 1] + lower_slack[step]
                >= conditions["comfort_min_c"].iloc[step]
            )
        else:
            constraints.append(
                temperature[step + 1] >= conditions["comfort_min_c"].iloc[step]
            )
        constraints.append(
            temperature[step + 1] <= conditions["comfort_max_c"].iloc[step]
        )

    energy_cost = dt * cp.sum(
        cp.multiply(conditions["import_price_gbp_per_kwh"].to_numpy(), electric_power)
    )
    slack_cost = slack_penalty * dt * cp.sum(lower_slack)
    objective = cp.Minimize(energy_cost + slack_cost)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Optimisation failed with status {problem.status}.")
    return {
        "status": problem.status,
        "objective": float(problem.value),
        "energy_cost_gbp": float(energy_cost.value),
        "slack_cost_gbp": float(slack_cost.value),
        "electric_power_kw": np.asarray(electric_power.value).ravel(),
        "predicted_temperature_c": np.asarray(temperature.value).ravel(),
        "comfort_slack_c": np.asarray(lower_slack.value).ravel(),
    }
""",
        """
def optimise_heating(
    conditions,
    model,
    initial_temperature_c=20.0,
    soft_constraints=True,
    slack_penalty=25.0,
):
    \"\"\"Return an optimal electrical-power and predicted-temperature schedule.\"\"\"
    n_steps = len(conditions)
    dt = model["sampling_interval_hours"]

    # TODO 1.1: create N nonnegative power and slack variables and N+1 temperatures.
    # TODO 1.2: add the initial condition, capacity and dynamics constraints.
    # TODO 1.3: use slack in the lower comfort bound only for soft constraints;
    #           for hard constraints, impose the bound without slack.
    # TODO 1.4: minimise energy cost plus the comfort-slack penalty.
    # TODO 1.5: solve, check problem.status and return costs and arrays.
    raise NotImplementedError("Formulate and solve the heating schedule.")
""",
    )
    measurement_experiment = choose(
        solution,
        """
def run_measurement_error_experiment(
    conditions,
    model,
    measurement_errors_c,
    soft_slack_penalty=0.25,
    true_initial_temperature_c=20.0,
):
    records = []
    for constraint_name, use_soft_constraints in {
        "Hard": False,
        "Soft": True,
    }.items():
        for measurement_error_c in measurement_errors_c:
            measured_initial_temperature = (
                true_initial_temperature_c + measurement_error_c
            )
            schedule = optimise_heating(
                conditions,
                model,
                initial_temperature_c=measured_initial_temperature,
                soft_constraints=use_soft_constraints,
                slack_penalty=soft_slack_penalty,
            )
            realised_temperature = simulate_matched_model(
                conditions,
                model,
                schedule["electric_power_kw"],
                initial_temperature_c=true_initial_temperature_c,
            )
            metrics = calculate_schedule_metrics(
                conditions,
                model,
                schedule["electric_power_kw"],
                realised_temperature,
            )
            records.append(
                {
                    "Constraint": constraint_name,
                    "Measurement error [°C]": measurement_error_c,
                    "Measured initial temperature [°C]": measured_initial_temperature,
                    "Predicted slack [°C h]": (
                        schedule["comfort_slack_c"].sum()
                        * model["sampling_interval_hours"]
                    ),
                    **metrics,
                }
            )

    return pd.DataFrame.from_records(records).set_index(
        ["Constraint", "Measurement error [°C]"]
    )
""",
        """
def run_measurement_error_experiment(
    conditions,
    model,
    measurement_errors_c,
    soft_slack_penalty=0.25,
    true_initial_temperature_c=20.0,
):
    \"\"\"Compare hard and soft schedules for signed initial measurement errors.\"\"\"
    # TODO 5.1: loop over hard and soft constraints and each measurement error.
    # TODO 5.2: optimise using measured temperature = true temperature + error.
    # TODO 5.3: simulate from the true temperature and calculate realised metrics.
    # TODO 5.4: return one tidy DataFrame containing every experiment.
    raise NotImplementedError("Run the measurement-error experiment.")
""",
    )
    return notebook(
        [
            md(
                """
# Tutorial 7.1: Optimisation foundations for building operation

This tutorial formulates a one-day heat-pump schedule using the verified $1R1C$ model. We compare hard and soft comfort constraints, investigate flat and dynamic electricity prices, verify the schedule in a perfectly matched model, and then introduce indoor-temperature measurement errors.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. distinguish optimisation variables from known parameters;
2. implement hard and soft operating constraints in CVXPY;
3. diagnose feasibility and verify a returned solution; and
4. explain how prices and initial-temperature measurement errors affect operation.

## Indicative schedule

- formulation and hard/soft constraints: 25 minutes;
- flat and dynamic tariff comparison: 15 minutes;
- matched-model checks: 15 minutes;
- measurement-error experiment: 20 minutes;
- interpretation and discussion: 15 minutes.
"""
            ),
            md("## 1. Set-up and one-day case"),
            setup_cell("import cvxpy as cp"),
            code(
                """
conditions = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_conditions.csv",
    index_col="timestamp",
    parse_dates=True,
).iloc[:48]
model = json.loads(
    (ROOT / "data" / "reference_building" / "verified_controller_model.json").read_text()
)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
axes[0].plot(conditions["outdoor_temperature_c"], label="Outdoor temperature")
axes[0].plot(conditions["comfort_min_c"], label="Lower comfort bound")
axes[0].plot(conditions["comfort_max_c"], label="Upper comfort bound")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].step(conditions.index, conditions["import_price_gbp_per_kwh"], where="post")
axes[1].set_ylabel("Import price [£/kWh]")
axes[1].set_xlabel("Time")
plt.show()
"""
            ),
            md(
                r"""
## 2. Task 1 — formulate the scheduling problem

The decision variables are electrical heat-pump power $P_{el,k}$, predicted temperature $T_k$ and lower comfort slack $s_k$. The model uses $q_{th,k}=\mathrm{COP}\,P_{el,k}$.

A hard comfort constraint must be satisfied exactly:

$$
T_k \geq T_k^{min}.
$$

A soft constraint introduces nonnegative slack $s_k$:

$$
T_k+s_k \geq T_k^{min}, \qquad s_k \geq 0.
$$

The soft-constraint objective is

$$
J=\sum_k p_kP_{el,k}\Delta t+\rho\sum_k s_k\Delta t.
$$

Hard constraints can make a problem infeasible. Soft constraints preserve feasibility when sufficient slack is allowed, but a low penalty $\rho$ can make avoidable discomfort economically attractive. Slack must therefore be penalised and reported rather than hidden.
"""
            ),
            code(optimiser),
            md(
                r"""
## 3. Task 2 — compare hard and soft constraints

The normal case is feasible with hard constraints. For the soft case, use $\rho=0.25$ £/(°C h) so that the energy–comfort trade-off is visible. This deliberately low teaching value is not a recommended operational setting.
"""
            ),
            code(
                """
true_initial_temperature_c = 20.0
soft_slack_penalty = 0.25

hard_schedule = optimise_heating(
    conditions,
    model,
    initial_temperature_c=true_initial_temperature_c,
    soft_constraints=False,
)
soft_schedule = optimise_heating(
    conditions,
    model,
    initial_temperature_c=true_initial_temperature_c,
    soft_constraints=True,
    slack_penalty=soft_slack_penalty,
)

schedule_comparison = pd.DataFrame(
    {
        "Hard": {
            "Energy cost [£]": hard_schedule["energy_cost_gbp"],
            "Slack penalty [£]": hard_schedule["slack_cost_gbp"],
            "Predicted slack [°C h]": hard_schedule["comfort_slack_c"].sum()
            * model["sampling_interval_hours"],
        },
        "Soft": {
            "Energy cost [£]": soft_schedule["energy_cost_gbp"],
            "Slack penalty [£]": soft_schedule["slack_cost_gbp"],
            "Predicted slack [°C h]": soft_schedule["comfort_slack_c"].sum()
            * model["sampling_interval_hours"],
        },
    }
).T
display(schedule_comparison.round(3))
"""
            ),
            code(
                """
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
axes[0].step(conditions.index, conditions["import_price_gbp_per_kwh"], where="post")
axes[0].set_ylabel("Price [£/kWh]")
axes[1].step(
    conditions.index,
    hard_schedule["electric_power_kw"],
    where="post",
    label="Hard",
)
axes[1].step(
    conditions.index,
    soft_schedule["electric_power_kw"],
    where="post",
    label="Soft",
    alpha=0.8,
)
axes[1].set_ylabel("Heat-pump power [kW]")
axes[1].legend()
axes[2].plot(
    conditions.index,
    hard_schedule["predicted_temperature_c"][1:],
    label="Hard",
)
axes[2].plot(
    conditions.index,
    soft_schedule["predicted_temperature_c"][1:],
    label="Soft",
)
axes[2].plot(conditions["comfort_min_c"], "--", label="Comfort bounds", color="black")
axes[2].plot(conditions["comfort_max_c"], "--", color="black")
axes[2].set_ylabel("Temperature [°C]")
axes[2].set_xlabel("Time")
axes[2].legend()
plt.show()
"""
            ),
            md(
                r"""
## 4. Task 3 — compare flat and dynamic prices

Create a flat tariff using the mean dynamic price. Everything else—the weather, comfort bounds, initial condition and hard constraints—remains unchanged. Under a flat price, the optimiser has no financial reason to move heating between intervals. Under the dynamic tariff, thermal storage allows it to preheat before expensive periods. Evaluate both schedules under both tariffs so that the scheduling effect is separated from the tariff definition.
"""
            ),
            code(
                """
flat_price_conditions = conditions.copy()
flat_price_conditions["import_price_gbp_per_kwh"] = conditions[
    "import_price_gbp_per_kwh"
].mean()

flat_price_schedule = optimise_heating(
    flat_price_conditions,
    model,
    initial_temperature_c=true_initial_temperature_c,
    soft_constraints=False,
)
dynamic_price_schedule = hard_schedule

tariff_comparison = pd.DataFrame(
    {
        "Flat tariff": {
            "Energy [kWh]": flat_price_schedule["electric_power_kw"].sum()
            * model["sampling_interval_hours"],
            "Cost under flat tariff [£]": flat_price_schedule["energy_cost_gbp"],
            "Cost under dynamic tariff [£]": float(
                np.sum(
                    flat_price_schedule["electric_power_kw"]
                    * conditions["import_price_gbp_per_kwh"].to_numpy()
                )
                * model["sampling_interval_hours"]
            ),
        },
        "Dynamic tariff": {
            "Energy [kWh]": dynamic_price_schedule["electric_power_kw"].sum()
            * model["sampling_interval_hours"],
            "Cost under flat tariff [£]": float(
                dynamic_price_schedule["electric_power_kw"].sum()
                * flat_price_conditions["import_price_gbp_per_kwh"].iloc[0]
                * model["sampling_interval_hours"]
            ),
            "Cost under dynamic tariff [£]": dynamic_price_schedule[
                "energy_cost_gbp"
            ],
        },
    }
).T
display(tariff_comparison.round(3))

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
axes[0].step(
    conditions.index,
    conditions["import_price_gbp_per_kwh"],
    where="post",
    label="Dynamic tariff",
)
axes[0].step(
    flat_price_conditions.index,
    flat_price_conditions["import_price_gbp_per_kwh"],
    where="post",
    label="Flat tariff",
)
axes[0].set_ylabel("Price [£/kWh]")
axes[0].legend()
axes[1].step(
    conditions.index,
    dynamic_price_schedule["electric_power_kw"],
    where="post",
    label="Dynamic tariff",
)
axes[1].step(
    conditions.index,
    flat_price_schedule["electric_power_kw"],
    where="post",
    label="Flat tariff",
    alpha=0.8,
)
axes[1].set_ylabel("Heat-pump power [kW]")
axes[1].legend()
axes[2].plot(
    conditions.index,
    dynamic_price_schedule["predicted_temperature_c"][1:],
    label="Dynamic tariff",
)
axes[2].plot(
    conditions.index,
    flat_price_schedule["predicted_temperature_c"][1:],
    label="Flat tariff",
)
axes[2].plot(conditions["comfort_min_c"], "--", color="black", label="Comfort bounds")
axes[2].plot(conditions["comfort_max_c"], "--", color="black")
axes[2].set_ylabel("Temperature [°C]")
axes[2].set_xlabel("Time")
axes[2].legend()
plt.show()
"""
            ),
            md(
                r"""
## 5. Task 4 — evaluate a perfectly matched model

We first remove every source of error. The controller model and evaluation model use exactly the same equation and parameters, the initial temperature is known exactly, and there is no process or measurement noise. Predicted and realised temperatures should therefore agree to numerical precision. This is a useful implementation check before studying uncertainty.
"""
            ),
            code(
                """
def simulate_matched_model(
    conditions,
    model,
    electric_power_kw,
    initial_temperature_c=20.0,
):
    \"\"\"Apply a fixed schedule using exactly the controller-model dynamics.\"\"\"
    temperatures = [float(initial_temperature_c)]
    for step, electric_power in enumerate(electric_power_kw):
        current_temperature = temperatures[-1]
        delivered_heat = model["nominal_cop"] * electric_power
        next_temperature = (
            current_temperature
            + model["loss_coefficient_per_step"]
            * (conditions["outdoor_temperature_c"].iloc[step] - current_temperature)
            + model["heat_coefficient_c_per_kw_step"] * delivered_heat
        )
        temperatures.append(next_temperature)
    return pd.Series(temperatures[1:], index=conditions.index, name="temperature_c")


def calculate_schedule_metrics(conditions, model, electric_power_kw, temperature_c):
    \"\"\"Return realised energy, cost and lower-comfort metrics.\"\"\"
    dt = model["sampling_interval_hours"]
    electric_power_kw = np.asarray(electric_power_kw, dtype=float)
    temperature_c = np.asarray(temperature_c, dtype=float)
    lower_bound = conditions["comfort_min_c"].to_numpy()
    violation = np.maximum(lower_bound - temperature_c, 0.0)
    return {
        "Energy [kWh]": float(electric_power_kw.sum() * dt),
        "Cost [£]": float(
            np.sum(
                electric_power_kw
                * conditions["import_price_gbp_per_kwh"].to_numpy()
            )
            * dt
        ),
        "Violation [h]": float(np.sum(violation > 1e-6) * dt),
        "Degree-hours [°C h]": float(violation.sum() * dt),
        "Maximum deviation [°C]": float(violation.max()),
    }
"""
            ),
            code(
                """
matched_results = {}
matched_metrics = {}
for constraint_name, schedule in {
    "Hard": hard_schedule,
    "Soft": soft_schedule,
}.items():
    realised_temperature = simulate_matched_model(
        conditions,
        model,
        schedule["electric_power_kw"],
        initial_temperature_c=true_initial_temperature_c,
    )
    matched_results[constraint_name] = realised_temperature
    predicted_temperature = schedule["predicted_temperature_c"][1:]
    maximum_mismatch = np.max(
        np.abs(realised_temperature.to_numpy() - predicted_temperature)
    )
    matched_metrics[constraint_name] = {
        **calculate_schedule_metrics(
            conditions,
            model,
            schedule["electric_power_kw"],
            realised_temperature,
        ),
        "Prediction mismatch [°C]": maximum_mismatch,
    }

matched_metric_table = pd.DataFrame(matched_metrics).T
display(matched_metric_table.round(4))
assert matched_metric_table["Prediction mismatch [°C]"].max() < 1e-8
"""
            ),
            md(
                r"""
## 6. Task 5 — sensitivity to measurement error

The true initial temperature remains 20 °C, but the optimiser receives

$$
T_0^{measured}=T_0^{true}+e_T.
$$

Use signed errors from −1 °C to +1 °C. A positive value means the sensor reads too warm, so the optimiser may schedule too little heat. Only the initial measurement enters this fixed open-loop schedule. Errors in later measurements would matter only if the controller recalculated its decisions, as in model predictive control.
"""
            ),
            code(measurement_experiment),
            code(
                """
measurement_errors_c = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
measurement_error_results = run_measurement_error_experiment(
    conditions,
    model,
    measurement_errors_c,
    soft_slack_penalty=soft_slack_penalty,
    true_initial_temperature_c=true_initial_temperature_c,
)
display(measurement_error_results.round(3))

plot_data = measurement_error_results.reset_index()
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for constraint_name, group in plot_data.groupby("Constraint"):
    group = group.sort_values("Measurement error [°C]")
    axes[0].plot(
        group["Measurement error [°C]"],
        group["Cost [£]"],
        marker="o",
        label=constraint_name,
    )
    axes[1].plot(
        group["Measurement error [°C]"],
        group["Degree-hours [°C h]"],
        marker="o",
        label=constraint_name,
    )

axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Initial measurement error [°C]")
axes[0].set_ylabel("Realised cost [£]")
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Initial measurement error [°C]")
axes[1].set_ylabel("Realised discomfort [°C h]")
axes[1].legend(title="Constraint")
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """
### Interpretation

Read the measurement-error axis as `measured minus true`. Compare both the magnitude and direction of the error. A colder reading tends to produce a conservative schedule with more heating, whereas a warmer reading can reduce cost while increasing realised comfort violations.

## Discussion

1. When does the dynamic-tariff optimiser preheat, and why?
2. What is gained and lost when a hard comfort constraint is replaced by a soft one?
3. Why are predicted and realised temperatures identical in the zero-error experiment?
4. Why does a positive initial-temperature measurement error tend to create more discomfort?
5. How does the slack penalty affect the balance between cost and comfort?
6. Why would repeated measurement errors matter more for a controller that re-optimises during the day?
"""
            ),
        ]
    )


def tutorial_7_2(solution):
    site_optimiser = choose(
        solution,
        """
def optimise_site(conditions, heat_pump_power_kw, peak_weight=0.35):
    n_steps = len(conditions)
    dt = 0.5
    eta_charge = 0.95
    eta_discharge = 0.95
    initial_energy = 5.0

    charge = cp.Variable(n_steps, nonneg=True)
    discharge = cp.Variable(n_steps, nonneg=True)
    stored_energy = cp.Variable(n_steps + 1)
    grid_import = cp.Variable(n_steps, nonneg=True)
    grid_export = cp.Variable(n_steps, nonneg=True)
    peak_import = cp.Variable(nonneg=True)

    constraints = [
        stored_energy[0] == initial_energy,
        stored_energy >= 0.5,
        stored_energy <= 9.5,
        charge <= 3.0,
        discharge <= 3.0,
        stored_energy[-1] >= initial_energy,
    ]
    for step in range(n_steps):
        constraints += [
            stored_energy[step + 1]
            == stored_energy[step]
            + dt * eta_charge * charge[step]
            - dt * discharge[step] / eta_discharge,
            grid_import[step]
            + conditions["pv_available_kw"].iloc[step]
            + discharge[step]
            == conditions["base_load_kw"].iloc[step]
            + heat_pump_power_kw[step]
            + charge[step]
            + grid_export[step],
            peak_import >= grid_import[step],
        ]

    cost = dt * cp.sum(
        cp.multiply(conditions["import_price_gbp_per_kwh"].to_numpy(), grid_import)
        - cp.multiply(conditions["export_price_gbp_per_kwh"].to_numpy(), grid_export)
    )
    problem = cp.Problem(cp.Minimize(cost + peak_weight * peak_import), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Site optimisation failed with status {problem.status}.")
    return pd.DataFrame(
        {
            "battery_charge_kw": np.asarray(charge.value).ravel(),
            "battery_discharge_kw": np.asarray(discharge.value).ravel(),
            "grid_import_kw": np.asarray(grid_import.value).ravel(),
            "grid_export_kw": np.asarray(grid_export.value).ravel(),
            "stored_energy_kwh": np.asarray(stored_energy.value[1:]).ravel(),
        },
        index=conditions.index,
    )
""",
        """
def optimise_site(conditions, heat_pump_power_kw, peak_weight=0.35):
    \"\"\"Schedule a 10 kWh battery around fixed building and heat-pump demand.\"\"\"
    n_steps = len(conditions)
    dt = 0.5

    # TODO 1.1: create charge, discharge, stored-energy, import, export and peak variables.
    # TODO 1.2: add battery energy dynamics, power limits and terminal energy.
    # TODO 1.3: enforce the site balance at every interval.
    # TODO 1.4: minimise import cost minus export revenue plus the peak term.
    # TODO 1.5: solve, check status and return the schedules as a DataFrame.
    raise NotImplementedError("Formulate the flexible-site optimisation.")
""",
    )
    return notebook(
        [
            md(
                """
# Tutorial 7.2: Dynamic optimisation and flexible building assets

This tutorial extends the heat-pump schedule with PV, base demand and a battery. The thermal schedule is kept fixed so that the new work remains focused on state-of-charge dynamics and the site electricity balance.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. formulate a battery energy-state equation;
2. enforce one consistent site electricity balance;
3. distinguish power from stored energy; and
4. compare cost, peak demand and event-period flexibility.

## Indicative schedule

- heat-pump schedule and site data: 15 minutes;
- battery and balance constraints: 35 minutes;
- objective and solution checks: 20 minutes;
- plots and flexibility metrics: 15 minutes;
- discussion: 5 minutes.
"""
            ),
            md("## 1. Set-up and fixed thermal schedule"),
            setup_cell("import cvxpy as cp\nfrom course_utils.control import solve_heating_schedule"),
            code(
                """
conditions = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_conditions.csv",
    index_col="timestamp",
    parse_dates=True,
).iloc[:48]
model = json.loads(
    (ROOT / "data" / "reference_building" / "verified_controller_model.json").read_text()
)
heating_solution = solve_heating_schedule(conditions, model)
heat_pump_power_kw = heating_solution["electric_power_kw"]

site_inputs = conditions[[
    "base_load_kw", "pv_available_kw", "import_price_gbp_per_kwh", "export_price_gbp_per_kwh"
]].copy()
site_inputs["heat_pump_power_kw"] = heat_pump_power_kw
display(site_inputs.head())
"""
            ),
            md(
                r"""
## 2. Task 1 — battery and site balance

For stored energy $E_k$, charge $P_k^{ch}$ and discharge $P_k^{dis}$:

$$
E_{k+1}=E_k+\eta_{ch}P_k^{ch}\Delta t-\frac{P_k^{dis}\Delta t}{\eta_{dis}}.
$$

One sign convention for the site balance is

$$
P_k^{imp}+P_k^{PV}+P_k^{dis}
=P_k^{base}+P_k^{HP}+P_k^{ch}+P_k^{exp}.
$$
"""
            ),
            code(site_optimiser),
            code(
                """
site_solution = optimise_site(conditions, heat_pump_power_kw)
display(site_solution.head())

balance_residual = (
    site_solution["grid_import_kw"]
    + conditions["pv_available_kw"]
    + site_solution["battery_discharge_kw"]
    - conditions["base_load_kw"]
    - heat_pump_power_kw
    - site_solution["battery_charge_kw"]
    - site_solution["grid_export_kw"]
)
print(f"Largest site-balance residual: {balance_residual.abs().max():.2e} kW")
print(f"Minimum stored energy: {site_solution['stored_energy_kwh'].min():.2f} kWh")
print(f"Maximum stored energy: {site_solution['stored_energy_kwh'].max():.2f} kWh")
"""
            ),
            md("## 3. Compare operation with and without storage"),
            code(
                """
net_without_storage = (
    conditions["base_load_kw"] + heat_pump_power_kw - conditions["pv_available_kw"]
)
import_without_storage = net_without_storage.clip(lower=0)
export_without_storage = (-net_without_storage).clip(lower=0)

dt = 0.5
cost_without_storage = dt * (
    (import_without_storage * conditions["import_price_gbp_per_kwh"]).sum()
    - (export_without_storage * conditions["export_price_gbp_per_kwh"]).sum()
)
cost_with_storage = dt * (
    (site_solution["grid_import_kw"] * conditions["import_price_gbp_per_kwh"]).sum()
    - (site_solution["grid_export_kw"] * conditions["export_price_gbp_per_kwh"]).sum()
)

comparison = pd.DataFrame(
    {
        "Cost [£]": [cost_without_storage, cost_with_storage],
        "Peak import [kW]": [import_without_storage.max(), site_solution["grid_import_kw"].max()],
        "Imported energy [kWh]": [import_without_storage.sum() * dt, site_solution["grid_import_kw"].sum() * dt],
    },
    index=["No battery", "Optimised battery"],
)
display(comparison.round(3))
"""
            ),
            code(
                """
fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
axes[0].plot(conditions["base_load_kw"], label="Base load")
axes[0].plot(conditions.index, heat_pump_power_kw, label="Heat pump")
axes[0].plot(conditions["pv_available_kw"], label="Available PV")
axes[0].set_ylabel("Power [kW]")
axes[0].legend(ncol=3)
axes[1].step(conditions.index, site_solution["battery_charge_kw"], where="post", label="Charge")
axes[1].step(conditions.index, -site_solution["battery_discharge_kw"], where="post", label="Discharge")
axes[1].set_ylabel("Battery [kW]")
axes[1].legend()
axes[2].plot(site_solution["stored_energy_kwh"])
axes[2].set_ylabel("Stored energy [kWh]")
axes[3].step(conditions.index, import_without_storage, where="post", label="No battery")
axes[3].step(conditions.index, site_solution["grid_import_kw"], where="post", label="Optimised battery")
axes[3].set_ylabel("Grid import [kW]")
axes[3].set_xlabel("Time")
axes[3].legend()
plt.show()
"""
            ),
            md("## 4. Event-period flexibility"),
            code(
                """
event_mask = (conditions.index.hour >= 16) & (conditions.index.hour < 19)
event_energy_without = float(import_without_storage.loc[event_mask].sum() * dt)
event_energy_with = float(site_solution.loc[event_mask, "grid_import_kw"].sum() * dt)
print(f"Event energy without battery: {event_energy_without:.2f} kWh")
print(f"Event energy with battery: {event_energy_with:.2f} kWh")
print(f"Event-period reduction: {event_energy_without - event_energy_with:.2f} kWh")
"""
            ),
            md(
                """
## Optional extension

Add an EV that is connected from 18:00 to 07:00. Give it a 7 kW charging limit and require a specified stored-energy increase by departure. Keep the EV availability schedule as a known parameter.

## Discussion

1. Why is a terminal battery-energy condition important?
2. Can the continuous formulation charge and discharge simultaneously? Did it do so here?
3. Why can reducing peak demand increase total cost?
4. Which asset data would need forecasts in real operation?
"""
            ),
        ]
    )


def tutorial_8_1(solution):
    mpc_loop = choose(
        solution,
        """
def run_mpc(
    operation,
    forecasts,
    model,
    evaluation_steps=48,
    horizon_steps=48,
    plant_class=TwoStateReferencePlant,
):
    controller = build_mpc_problem(model, horizon_steps)
    plant = plant_class(
        step_hours=model["sampling_interval_hours"],
        random_state=88,
    )
    measured_temperature = plant.reset(20.0, 20.0)
    records = []

    for position in range(evaluation_steps):
        timestamp = operation.index[position]
        horizon = operation.iloc[position : position + horizon_steps]
        outdoor_forecast = forecast_vector(
            forecasts,
            timestamp,
            "outdoor_temperature_forecast_c",
            operation["outdoor_temperature_c"].iloc[position],
            horizon_steps,
        )

        controller["initial_temperature"].value = measured_temperature
        controller["outdoor_temperature"].value = outdoor_forecast
        controller["price"].value = horizon["import_price_gbp_per_kwh"].to_numpy()
        controller["comfort_min"].value = horizon["comfort_min_c"].to_numpy()
        controller["comfort_max"].value = horizon["comfort_max_c"].to_numpy()
        controller["problem"].solve(solver=cp.CLARABEL, warm_start=True)

        status = controller["problem"].status
        fallback = status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if fallback:
            action_kw = (
                model["heat_pump_max_electric_kw"]
                if measured_temperature < operation["comfort_min_c"].iloc[position]
                else 0.0
            )
            predicted_next = np.nan
        else:
            action_kw = float(controller["electric_power"].value[0])
            predicted_next = float(controller["temperature"].value[1])

        realised = operation.iloc[position]
        result = plant.step(
            action_kw / model["heat_pump_max_electric_kw"],
            realised["outdoor_temperature_c"],
            realised["solar_irradiance_w_m2"],
            realised["internal_gains_kw"],
        )
        records.append(
            {
                "timestamp": timestamp,
                "indoor_temperature_c": measured_temperature,
                "next_indoor_temperature_c": result["indoor_temperature_measured_c"],
                "predicted_next_temperature_c": predicted_next,
                "heat_pump_electric_kw": result["heat_pump_electric_kw"],
                "solver_status": status,
                "fallback": fallback,
            }
        )
        measured_temperature = result["indoor_temperature_measured_c"]

    return pd.DataFrame.from_records(records).set_index("timestamp")
""",
        """
def run_mpc(
    operation,
    forecasts,
    model,
    evaluation_steps=48,
    horizon_steps=48,
    plant_class=TwoStateReferencePlant,
):
    \"\"\"Run a receding-horizon controller and return the realised log.\"\"\"
    controller = build_mpc_problem(model, horizon_steps)
    plant = plant_class(
        step_hours=model["sampling_interval_hours"],
        random_state=88,
    )
    measured_temperature = plant.reset(20.0, 20.0)
    records = []

    # TODO 1.1: loop over the evaluation intervals.
    # TODO 1.2: slice an origin-valid forecast and update every CVXPY parameter.
    # TODO 1.3: solve, check status and select only the first action.
    # TODO 1.4: advance the reference plant using realised disturbances.
    # TODO 1.5: log the transition and update the measured temperature.
    raise NotImplementedError("Complete the receding-horizon loop.")
""",
    )
    return notebook(
        [
            md(
                """
# Tutorial 8.1: Model predictive control for buildings

This tutorial converts the Week 7 fixed schedule into a receding-horizon controller. At every update, the current measured temperature anchors a new prediction, only the first optimised action is applied, and the reference plant advances under realised conditions.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. implement a receding-horizon control loop;
2. align forecasts, actions and state transitions;
3. apply only the first action of each optimal sequence; and
4. compare MPC with a thermostat and a fixed schedule using common summary metrics.

## Indicative schedule

- one MPC update and indexing: 20 minutes;
- receding-horizon loop: 35 minutes;
- solver checks and fallback: 15 minutes;
- controller comparison: 15 minutes;
- discussion: 5 minutes.
"""
            ),
            md("## 1. Set-up, data and forecasts"),
            setup_cell(
                "import cvxpy as cp\nfrom course_utils.reference_building import TwoStateReferencePlant\nfrom course_utils.mpc import build_mpc_problem, forecast_vector\nfrom course_utils.control import solve_heating_schedule, simulate_reference_schedule, simulate_thermostat, realised_metrics"
            ),
            code(
                """
operation = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_conditions.csv",
    index_col="timestamp",
    parse_dates=True,
).sort_index()
forecasts = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_forecasts.csv",
    parse_dates=["issue_time", "valid_time"],
)
model = json.loads(
    (ROOT / "data" / "reference_building" / "verified_controller_model.json").read_text()
)

evaluation_conditions = operation.iloc[:48]
first_origin = evaluation_conditions.index[0]
first_forecast = forecasts.loc[forecasts["issue_time"] == first_origin].head(47)
print(f"Forecast origin: {first_origin}")
display(first_forecast.head())
"""
            ),
            md(
                """
## 2. One update before the full loop

At update $t$, the current measured state becomes $T_{0|t}$. The optimiser proposes $N$ actions but the plant receives only $P^*_{0|t}$. At the next interval, the remaining proposal is discarded.
"""
            ),
            code(
                """
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

display_horizon = 8
updates_to_show = 5
timeline_columns = display_horizon + updates_to_show - 1

fixed_plan = np.ones((1, timeline_columns))
rolling_plan = np.zeros((updates_to_show, timeline_columns))
for update in range(updates_to_show):
    rolling_plan[update, update : update + display_horizon] = 1
    rolling_plan[update, update] = 2

colour_map = ListedColormap(["white", "#4C78A8", "#F58518"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1, 1.7]})
axes[0].imshow(fixed_plan, cmap=colour_map, vmin=0, vmax=2, aspect="auto")
axes[0].set_yticks([0], ["Optimise once"])
axes[0].set_title("Fixed open-loop schedule")
axes[1].imshow(rolling_plan, cmap=colour_map, vmin=0, vmax=2, aspect="auto")
axes[1].set_yticks(range(updates_to_show), [f"Update {i}" for i in range(updates_to_show)])
axes[1].set_title("Receding horizon")
for axis in axes:
    axis.set_xticks(range(timeline_columns))
    axis.set_xlabel("Control interval")
    axis.set_xticks(np.arange(-0.5, timeline_columns, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, axis.images[0].get_array().shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.5)
    axis.tick_params(which="minor", bottom=False, left=False)

axes[1].legend(
    handles=[
        Patch(color="#4C78A8", label="Planned, then reconsidered"),
        Patch(color="#F58518", label="Applied now"),
    ],
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
)
plt.tight_layout()
plt.show()
"""
            ),
            code(
                """
horizon_steps = 48
example_controller = build_mpc_problem(model, horizon_steps)
example_outdoor = forecast_vector(
    forecasts,
    first_origin,
    "outdoor_temperature_forecast_c",
    operation["outdoor_temperature_c"].iloc[0],
    horizon_steps,
)
example_horizon = operation.iloc[:horizon_steps]
example_controller["initial_temperature"].value = 20.0
example_controller["outdoor_temperature"].value = example_outdoor
example_controller["price"].value = example_horizon["import_price_gbp_per_kwh"].to_numpy()
example_controller["comfort_min"].value = example_horizon["comfort_min_c"].to_numpy()
example_controller["comfort_max"].value = example_horizon["comfort_max_c"].to_numpy()
example_controller["problem"].solve(solver=cp.CLARABEL)

print("Status:", example_controller["problem"].status)
print(f"First action applied: {example_controller['electric_power'].value[0]:.3f} kW")
print("The other 47 proposed actions will be reconsidered at the next update.")
"""
            ),
            md("## 3. Task 1 — implement the receding-horizon loop"),
            code(mpc_loop),
            code(
                """
mpc_results = run_mpc(operation, forecasts, model)
print(mpc_results["solver_status"].value_counts())
print(f"Fallback actions: {int(mpc_results['fallback'].sum())}")
display(mpc_results.head())
"""
            ),
            md(
                """
## 4. Compare three strategies on the same plant

Plots show when each controller acts, but they are not sufficient for comparison. Report total electrical energy, total energy cost, peak power, time outside the lower comfort bound, degree-hours of discomfort and maximum temperature deviation for every strategy.
"""
            ),
            code(
                """
fixed_solution = solve_heating_schedule(evaluation_conditions, model)
fixed_results = simulate_reference_schedule(
    evaluation_conditions,
    fixed_solution["electric_power_kw"],
    random_state=88,
)
thermostat_results = simulate_thermostat(evaluation_conditions, random_state=88)

comparison = pd.DataFrame(
    {
        "Thermostat": realised_metrics(thermostat_results, evaluation_conditions),
        "Fixed schedule": realised_metrics(fixed_results, evaluation_conditions),
        "MPC": realised_metrics(mpc_results, evaluation_conditions),
    }
).T
display(comparison.round(3))

relative_to_thermostat = pd.DataFrame(
    {
        "Energy change [%]": 100
        * (comparison["Energy [kWh]"] / comparison.loc["Thermostat", "Energy [kWh]"] - 1),
        "Cost change [%]": 100
        * (comparison["Cost [£]"] / comparison.loc["Thermostat", "Cost [£]"] - 1),
    }
)
display(relative_to_thermostat.round(1))
"""
            ),
            code(
                """
fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9))
axes[0].plot(mpc_results["next_indoor_temperature_c"], label="MPC")
axes[0].plot(fixed_results["next_indoor_temperature_c"], label="Fixed schedule")
axes[0].plot(thermostat_results["next_indoor_temperature_c"], label="Thermostat")
axes[0].plot(evaluation_conditions["comfort_min_c"], "--", color="black", label="Comfort bounds")
axes[0].plot(evaluation_conditions["comfort_max_c"], "--", color="black")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend(ncol=2)
axes[1].step(mpc_results.index, mpc_results["heat_pump_electric_kw"], where="post", label="MPC")
axes[1].step(fixed_results.index, fixed_results["heat_pump_electric_kw"], where="post", label="Fixed")
axes[1].step(thermostat_results.index, thermostat_results["heat_pump_electric_kw"], where="post", label="Thermostat", alpha=0.7)
axes[1].set_ylabel("Heat-pump power [kW]")
axes[1].legend()
axes[2].step(evaluation_conditions.index, evaluation_conditions["import_price_gbp_per_kwh"], where="post")
axes[2].set_ylabel("Price [£/kWh]")
axes[2].set_xlabel("Time")
plt.show()
"""
            ),
            md(
                """
## Discussion

1. Which information changes between two MPC updates?
2. Why can MPC correct accumulated state error while a fixed schedule cannot?
3. What errors remain until the next measurement?
4. Why must a safe fallback be defined before running the controller?
"""
            ),
        ]
    )


def tutorial_8_2(solution):
    metrics_function = choose(
        solution,
        """
def controller_metrics(results, conditions):
    power = results["heat_pump_electric_kw"].to_numpy()
    temperature = results["next_indoor_temperature_c"].to_numpy()
    lower = conditions["comfort_min_c"].to_numpy()
    violation = np.maximum(lower - temperature, 0.0)
    dt = 0.5
    return {
        "Energy [kWh]": float(power.sum() * dt),
        "Cost [£]": float(np.sum(power * conditions["import_price_gbp_per_kwh"].to_numpy()) * dt),
        "Peak [kW]": float(power.max()),
        "Violation [h]": float(np.sum(violation > 0) * dt),
        "Degree-hours [°C h]": float(violation.sum() * dt),
        "Maximum deviation [°C]": float(violation.max()),
        "Mean solve time [s]": float(results["solve_time_s"].mean()),
        "Fallback count": int(results["fallback"].sum()),
    }
""",
        """
def controller_metrics(results, conditions):
    \"\"\"Calculate realised energy, cost, peak, comfort and implementation metrics.\"\"\"
    # TODO 1.1: calculate energy and cost from realised electrical power.
    # TODO 1.2: calculate violation time, degree-hours and maximum deviation.
    # TODO 1.3: add mean solve time and fallback count.
    raise NotImplementedError("Calculate the realised controller metrics.")
""",
    )
    scenarios_code = choose(
        solution,
        """
scenarios = {
    "Nominal": {},
    "Cold-biased forecast": {"forecast_temperature_bias_c": -1.5},
    "Higher heat loss": {"plant_resistance_multiplier": 0.80},
    "Higher thermal mass": {"plant_capacitance_multiplier": 1.30},
    "Warm sensor bias": {"sensor_bias_c": 0.40},
    "Tightened comfort band": {"comfort_margin_c": 0.30},
}
""",
        """
# TODO 2: define the six scenarios described above as keyword-argument dictionaries.
scenarios = {
    "Nominal": {},
}
""",
    )
    return notebook(
        [
            md(
                """
# Tutorial 8.2: MPC under uncertainty and controller evaluation

This tutorial evaluates the same MPC formulation under forecast, model and measurement errors. All scenarios use the same reporting period and metric definitions. We also compare MPC with a fixed open-loop schedule to show the value of feedback under uncertainty.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. distinguish forecast, model and measurement errors;
2. calculate ex-post operational metrics from the reference plant;
3. compare fixed and feedback controllers under a common realised scenario; and
4. explain the benefits and limits of feedback and constraint tightening.

## Indicative schedule

- experiment design and metrics: 20 minutes;
- scenario runs: 30 minutes;
- fixed versus feedback control: 20 minutes;
- robustness and sensitivity plots: 15 minutes;
- interpretation: 10 minutes;
- discussion: 5 minutes.
"""
            ),
            md("## 1. Set-up and common evaluation period"),
            setup_cell(
                "from course_utils.mpc import run_fixed_schedule_experiment, run_mpc_experiment"
            ),
            code(
                """
operation = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_conditions.csv",
    index_col="timestamp",
    parse_dates=True,
).sort_index()
forecasts = pd.read_csv(
    ROOT / "data" / "reference_building" / "operation_forecasts.csv",
    parse_dates=["issue_time", "valid_time"],
)
model = json.loads(
    (ROOT / "data" / "reference_building" / "verified_controller_model.json").read_text()
)
evaluation_conditions = operation.iloc[:48]
"""
            ),
            md(
                """
## 2. Task 1 — realised metrics

Ex-ante quantities come from the controller model before an action is applied. Ex-post metrics use realised reference-plant temperature and power. Controller claims should use the latter.
"""
            ),
            code(metrics_function),
            md(
                """
## 3. Task 2 — define uncertainty scenarios

Use one nominal case, a cold-biased forecast, a plant with higher heat loss, a plant with higher thermal mass, a warm sensor bias and a tightened comfort band. Each scenario changes one assumption while holding the reporting period and controller structure fixed.
"""
            ),
            code(scenarios_code),
            code(
                """
scenario_results = {}
scenario_metrics = {}
for name, arguments in scenarios.items():
    result = run_mpc_experiment(
        operation,
        forecasts,
        model,
        evaluation_steps=48,
        horizon_steps=48,
        random_state=90,
        **arguments,
    )
    scenario_results[name] = result
    scenario_metrics[name] = controller_metrics(result, evaluation_conditions)

metric_table = pd.DataFrame(scenario_metrics).T
display(metric_table.round(3))
"""
            ),
            md("## 4. Compare realised trajectories"),
            code(
                """
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
for name in ["Nominal", "Cold-biased forecast", "Higher heat loss", "Warm sensor bias"]:
    if name in scenario_results:
        axes[0].plot(
            scenario_results[name]["next_indoor_temperature_c"],
            label=name,
        )
axes[0].plot(evaluation_conditions["comfort_min_c"], "--", color="black", label="Comfort bounds")
axes[0].plot(evaluation_conditions["comfort_max_c"], "--", color="black")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend(ncol=2)
for name in ["Nominal", "Cold-biased forecast", "Higher heat loss", "Warm sensor bias"]:
    if name in scenario_results:
        axes[1].step(
            scenario_results[name].index,
            scenario_results[name]["heat_pump_electric_kw"],
            where="post",
            label=name,
        )
axes[1].set_ylabel("Heat-pump power [kW]")
axes[1].set_xlabel("Time")
axes[1].legend(ncol=2)
plt.show()
"""
            ),
            md(
                """
## 5. Task 3 — fixed schedule versus MPC under uncertainty

A fixed controller optimises once and cannot react when the realised building departs from its model. MPC receives a new indoor-temperature measurement every 30 minutes and can correct the remaining schedule.

Use a deliberately severe heat-loss error as a stress test: the reference plant's thermal resistance is half the assumed value. Both controllers face the same weather, plant parameters, noise sequence and reporting period. The purpose is to expose the feedback mechanism, not to claim that such a large parameter error is typical.
"""
            ),
            code(
                """
stress_scenario = {"plant_resistance_multiplier": 0.50}

fixed_stress_results = run_fixed_schedule_experiment(
    operation,
    forecasts,
    model,
    evaluation_steps=48,
    horizon_steps=48,
    random_state=90,
    **stress_scenario,
)
mpc_stress_results = run_mpc_experiment(
    operation,
    forecasts,
    model,
    evaluation_steps=48,
    horizon_steps=48,
    random_state=90,
    **stress_scenario,
)

feedback_comparison = pd.DataFrame(
    {
        "Fixed open-loop schedule": controller_metrics(
            fixed_stress_results, evaluation_conditions
        ),
        "MPC with feedback": controller_metrics(
            mpc_stress_results, evaluation_conditions
        ),
    }
).T
display(feedback_comparison.round(3))

discomfort_reduction_percent = 100 * (
    1
    - feedback_comparison.loc["MPC with feedback", "Degree-hours [°C h]"]
    / feedback_comparison.loc[
        "Fixed open-loop schedule", "Degree-hours [°C h]"
    ]
)
print(
    "MPC reduction in degree-hours relative to the fixed schedule: "
    f"{discomfort_reduction_percent:.1f}%"
)
assert (
    feedback_comparison.loc["MPC with feedback", "Degree-hours [°C h]"]
    < feedback_comparison.loc[
        "Fixed open-loop schedule", "Degree-hours [°C h]"
    ]
)
"""
            ),
            code(
                """
fig, axes = plt.subplots(3, 1, figsize=(10, 9))
axes[0].plot(
    fixed_stress_results["next_indoor_temperature_c"],
    label="Fixed open-loop schedule",
)
axes[0].plot(
    mpc_stress_results["next_indoor_temperature_c"],
    label="MPC with feedback",
)
axes[0].plot(
    evaluation_conditions["comfort_min_c"],
    "--",
    color="black",
    label="Comfort bounds",
)
axes[0].plot(evaluation_conditions["comfort_max_c"], "--", color="black")
axes[0].set_ylabel("Temperature [°C]")
axes[0].legend()
axes[1].step(
    fixed_stress_results.index,
    fixed_stress_results["heat_pump_electric_kw"],
    where="post",
    label="Fixed open-loop schedule",
)
axes[1].step(
    mpc_stress_results.index,
    mpc_stress_results["heat_pump_electric_kw"],
    where="post",
    label="MPC with feedback",
)
axes[1].set_ylabel("Heat-pump power [kW]")
axes[1].legend()
axes[2].bar(
    feedback_comparison.index,
    feedback_comparison["Degree-hours [°C h]"],
)
axes[2].set_ylabel("Discomfort [°C h]")
axes[2].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """
The feedback controller uses more energy in this stress test, but it substantially reduces accumulated discomfort and maximum temperature deviation. This is the central comparison: feedback does not remove uncertainty, but it prevents state error from remaining uncorrected throughout the full schedule.

## 6. Prediction-horizon sensitivity

A longer horizon can anticipate prices and occupancy earlier, but it relies on more distant forecasts and takes more time to solve.
"""
            ),
            code(
                """
horizon_metrics = {}
for horizon_steps in [12, 24, 48]:
    result = run_mpc_experiment(
        operation,
        forecasts,
        model,
        evaluation_steps=48,
        horizon_steps=horizon_steps,
        random_state=90,
    )
    horizon_metrics[f"{horizon_steps / 2:.0f} h"] = controller_metrics(
        result, evaluation_conditions
    )

horizon_table = pd.DataFrame(horizon_metrics).T
display(horizon_table.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
horizon_table["Cost [£]"].plot.bar(ax=axes[0], title="Realised cost")
horizon_table["Degree-hours [°C h]"].plot.bar(ax=axes[1], title="Realised discomfort")
axes[0].set_ylabel("Cost [£]")
axes[1].set_ylabel("Degree-hours [°C h]")
plt.tight_layout()
plt.show()
"""
            ),
            md(
                """
## Discussion

1. Which error source has the largest operational effect in this experiment?
2. Does the cold-biased forecast necessarily worsen comfort?
3. Why does the fixed schedule accumulate more discomfort in the severe heat-loss case?
4. What does MPC trade for its improvement in comfort?
5. What does constraint tightening trade against improved robustness?
6. Why can feedback fail to correct an error immediately?
7. Why should controller tuning scenarios remain separate from final evaluation scenarios?
"""
            ),
        ]
    )


def write_pair(stem, builder):
    student = builder(False)
    solution = builder(True)
    nbf.write(student, ROOT / f"{stem}.ipynb")
    nbf.write(solution, ROOT / f"{stem}_solution.ipynb")


def main():
    write_pair("tutorial_6_1_control_oriented_building_model", tutorial_6_1)
    write_pair("tutorial_6_2_rc_model_identification", tutorial_6_2)
    write_pair("tutorial_7_1_building_optimisation", tutorial_7_1)
    write_pair("tutorial_7_2_flexible_building_assets", tutorial_7_2)
    write_pair("tutorial_8_1_model_predictive_control", tutorial_8_1)
    write_pair("tutorial_8_2_mpc_uncertainty_evaluation", tutorial_8_2)


if __name__ == "__main__":
    main()
