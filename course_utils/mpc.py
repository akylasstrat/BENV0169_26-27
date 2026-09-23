"""Reusable linear MPC experiment used in Tutorials 8.1 and 8.2."""

from dataclasses import replace
from time import perf_counter

import cvxpy as cp
import numpy as np
import pandas as pd

from course_utils.reference_building import (
    ReferenceBuildingParameters,
    TwoStateReferencePlant,
)


def build_mpc_problem(model, horizon_steps, slack_penalty=25.0):
    """Build one parameterised linear MPC problem."""

    n = int(horizon_steps)
    electric_power = cp.Variable(n, nonneg=True)
    temperature = cp.Variable(n + 1)
    lower_slack = cp.Variable(n, nonneg=True)

    initial_temperature = cp.Parameter()
    outdoor_temperature = cp.Parameter(n)
    price = cp.Parameter(n, nonneg=True)
    comfort_min = cp.Parameter(n)
    comfort_max = cp.Parameter(n)

    constraints = [
        temperature[0] == initial_temperature,
        electric_power <= model["heat_pump_max_electric_kw"],
    ]
    for step in range(n):
        constraints += [
            temperature[step + 1]
            == temperature[step]
            + model["loss_coefficient_per_step"]
            * (outdoor_temperature[step] - temperature[step])
            + model["heat_coefficient_c_per_kw_step"]
            * model["nominal_cop"]
            * electric_power[step],
            temperature[step + 1] + lower_slack[step] >= comfort_min[step],
            temperature[step + 1] <= comfort_max[step],
        ]

    dt = model["sampling_interval_hours"]
    objective = cp.Minimize(
        dt * cp.sum(cp.multiply(price, electric_power))
        + slack_penalty * dt * cp.sum(lower_slack)
    )
    problem = cp.Problem(objective, constraints)
    return {
        "problem": problem,
        "electric_power": electric_power,
        "temperature": temperature,
        "lower_slack": lower_slack,
        "initial_temperature": initial_temperature,
        "outdoor_temperature": outdoor_temperature,
        "price": price,
        "comfort_min": comfort_min,
        "comfort_max": comfort_max,
    }


def forecast_vector(forecasts, issue_time, column, current_value, horizon_steps):
    """Return the current measured value followed by origin-valid forecasts."""

    rows = forecasts.loc[forecasts["issue_time"] == issue_time].sort_values("lead_steps")
    future = rows[column].to_numpy()[: horizon_steps - 1]
    if len(future) != horizon_steps - 1:
        raise ValueError(f"Insufficient forecast values at issue time {issue_time}.")
    return np.concatenate([[float(current_value)], future])


def run_fixed_schedule_experiment(
    operation,
    forecasts,
    model,
    evaluation_steps=48,
    horizon_steps=48,
    forecast_temperature_bias_c=0.0,
    plant_resistance_multiplier=1.0,
    plant_capacitance_multiplier=1.0,
    plant_cop_multiplier=1.0,
    sensor_bias_c=0.0,
    comfort_margin_c=0.0,
    random_state=90,
    plant_class=TwoStateReferencePlant,
):
    """Optimise once, then apply the fixed schedule to one plant scenario."""

    if evaluation_steps > horizon_steps:
        raise ValueError("The fixed optimisation horizon must cover the evaluation period.")
    if horizon_steps > len(operation):
        raise ValueError("Operation data must cover the fixed optimisation horizon.")

    controller = build_mpc_problem(model, horizon_steps)
    base_parameters = ReferenceBuildingParameters()
    plant_parameters = replace(
        base_parameters,
        resistance_outdoor_fabric_k_per_kw=(
            base_parameters.resistance_outdoor_fabric_k_per_kw
            * plant_resistance_multiplier
        ),
        capacitance_fabric_kwh_per_k=(
            base_parameters.capacitance_fabric_kwh_per_k
            * plant_capacitance_multiplier
        ),
        nominal_cop=base_parameters.nominal_cop * plant_cop_multiplier,
    )
    plant = plant_class(
        parameters=plant_parameters,
        step_hours=model["sampling_interval_hours"],
        random_state=random_state,
    )
    measured_temperature = plant.reset(20.0, 20.0)

    first_origin = operation.index[0]
    horizon = operation.iloc[:horizon_steps]
    outdoor_forecast = forecast_vector(
        forecasts,
        first_origin,
        "outdoor_temperature_forecast_c",
        operation["outdoor_temperature_c"].iloc[0],
        horizon_steps,
    ) + forecast_temperature_bias_c

    controller["initial_temperature"].value = measured_temperature + sensor_bias_c
    controller["outdoor_temperature"].value = outdoor_forecast
    controller["price"].value = horizon["import_price_gbp_per_kwh"].to_numpy()
    controller["comfort_min"].value = (
        horizon["comfort_min_c"].to_numpy() + comfort_margin_c
    )
    controller["comfort_max"].value = (
        horizon["comfort_max_c"].to_numpy() - comfort_margin_c
    )

    start = perf_counter()
    controller["problem"].solve(solver=cp.CLARABEL)
    solve_time = perf_counter() - start
    status = controller["problem"].status
    fallback = status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if fallback:
        fixed_power = np.zeros(evaluation_steps)
        predicted_temperature = np.full(evaluation_steps, np.nan)
    else:
        fixed_power = np.clip(
            np.asarray(controller["electric_power"].value).ravel()[:evaluation_steps],
            0,
            model["heat_pump_max_electric_kw"],
        )
        predicted_temperature = np.asarray(controller["temperature"].value).ravel()[
            1 : evaluation_steps + 1
        ]

    records = []
    for position in range(evaluation_steps):
        timestamp = operation.index[position]
        row = operation.iloc[position]
        result = plant.step(
            fixed_power[position] / model["heat_pump_max_electric_kw"],
            row["outdoor_temperature_c"],
            row["solar_irradiance_w_m2"],
            row["internal_gains_kw"],
        )
        records.append(
            {
                "timestamp": timestamp,
                "indoor_temperature_c": measured_temperature,
                "controller_observation_c": (
                    measured_temperature + sensor_bias_c if position == 0 else np.nan
                ),
                "next_indoor_temperature_c": result["indoor_temperature_measured_c"],
                "heat_pump_electric_kw": result["heat_pump_electric_kw"],
                "predicted_next_temperature_c": predicted_temperature[position],
                "solver_status": status,
                "fallback": fallback,
                "solve_time_s": solve_time if position == 0 else np.nan,
            }
        )
        measured_temperature = result["indoor_temperature_measured_c"]

    return pd.DataFrame.from_records(records).set_index("timestamp")


def run_mpc_experiment(
    operation,
    forecasts,
    model,
    evaluation_steps=48,
    horizon_steps=48,
    forecast_temperature_bias_c=0.0,
    plant_resistance_multiplier=1.0,
    plant_capacitance_multiplier=1.0,
    plant_cop_multiplier=1.0,
    sensor_bias_c=0.0,
    comfort_margin_c=0.0,
    random_state=90,
    plant_class=TwoStateReferencePlant,
):
    """Run one closed-loop MPC scenario on the independent reference plant."""

    if evaluation_steps + horizon_steps > len(operation):
        raise ValueError("Operation data must cover the evaluation and prediction horizon.")
    controller = build_mpc_problem(model, horizon_steps)

    base_parameters = ReferenceBuildingParameters()
    plant_parameters = replace(
        base_parameters,
        resistance_outdoor_fabric_k_per_kw=(
            base_parameters.resistance_outdoor_fabric_k_per_kw * plant_resistance_multiplier
        ),
        capacitance_fabric_kwh_per_k=(
            base_parameters.capacitance_fabric_kwh_per_k * plant_capacitance_multiplier
        ),
        nominal_cop=base_parameters.nominal_cop * plant_cop_multiplier,
    )
    plant = plant_class(
        parameters=plant_parameters,
        step_hours=model["sampling_interval_hours"],
        random_state=random_state,
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
        ) + forecast_temperature_bias_c

        controller["initial_temperature"].value = measured_temperature + sensor_bias_c
        controller["outdoor_temperature"].value = outdoor_forecast
        controller["price"].value = horizon["import_price_gbp_per_kwh"].to_numpy()
        controller["comfort_min"].value = horizon["comfort_min_c"].to_numpy() + comfort_margin_c
        controller["comfort_max"].value = horizon["comfort_max_c"].to_numpy() - comfort_margin_c

        start = perf_counter()
        controller["problem"].solve(solver=cp.CLARABEL, warm_start=True)
        solve_time = perf_counter() - start
        status = controller["problem"].status
        fallback = status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
        if fallback:
            action_kw = model["heat_pump_max_electric_kw"] if measured_temperature < operation["comfort_min_c"].iloc[position] else 0.0
        else:
            action_kw = float(np.clip(controller["electric_power"].value[0], 0, model["heat_pump_max_electric_kw"]))

        row = operation.iloc[position]
        result = plant.step(
            action_kw / model["heat_pump_max_electric_kw"],
            row["outdoor_temperature_c"],
            row["solar_irradiance_w_m2"],
            row["internal_gains_kw"],
        )
        records.append(
            {
                "timestamp": timestamp,
                "indoor_temperature_c": measured_temperature,
                "controller_observation_c": measured_temperature + sensor_bias_c,
                "next_indoor_temperature_c": result["indoor_temperature_measured_c"],
                "heat_pump_electric_kw": result["heat_pump_electric_kw"],
                "predicted_next_temperature_c": (
                    np.nan if fallback else float(controller["temperature"].value[1])
                ),
                "solver_status": status,
                "fallback": fallback,
                "solve_time_s": solve_time,
            }
        )
        measured_temperature = result["indoor_temperature_measured_c"]

    return pd.DataFrame.from_records(records).set_index("timestamp")
