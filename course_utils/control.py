"""Small control helpers shared by the Weeks 7--8 teaching notebooks."""

import cvxpy as cp
import numpy as np
import pandas as pd

from course_utils.reference_building import TwoStateReferencePlant


def solve_heating_schedule(
    conditions,
    model,
    initial_temperature_c=20.0,
    slack_penalty=25.0,
    peak_weight=0.0,
):
    """Solve the linear heat-pump scheduling problem for one fixed horizon."""

    n_steps = len(conditions)
    dt = float(model["sampling_interval_hours"])
    b_loss = float(model["loss_coefficient_per_step"])
    b_heat = float(model["heat_coefficient_c_per_kw_step"])
    cop = float(model["nominal_cop"])
    maximum_power = float(model["heat_pump_max_electric_kw"])

    electric_power = cp.Variable(n_steps, nonneg=True)
    temperature = cp.Variable(n_steps + 1)
    lower_slack = cp.Variable(n_steps, nonneg=True)
    peak_power = cp.Variable(nonneg=True)

    constraints = [temperature[0] == initial_temperature_c, electric_power <= maximum_power]
    for step in range(n_steps):
        thermal_power = cop * electric_power[step]
        constraints += [
            temperature[step + 1]
            == temperature[step]
            + b_loss * (conditions["outdoor_temperature_c"].iloc[step] - temperature[step])
            + b_heat * thermal_power,
            temperature[step + 1] + lower_slack[step] >= conditions["comfort_min_c"].iloc[step],
            temperature[step + 1] <= conditions["comfort_max_c"].iloc[step],
            peak_power >= electric_power[step],
        ]

    energy_cost = dt * cp.sum(
        cp.multiply(conditions["import_price_gbp_per_kwh"].to_numpy(), electric_power)
    )
    objective = cp.Minimize(energy_cost + slack_penalty * dt * cp.sum(lower_slack) + peak_weight * peak_power)
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Heating optimisation failed with status {problem.status}.")

    return {
        "status": problem.status,
        "objective": float(problem.value),
        "electric_power_kw": np.asarray(electric_power.value).ravel(),
        "predicted_temperature_c": np.asarray(temperature.value).ravel(),
        "comfort_slack_c": np.asarray(lower_slack.value).ravel(),
    }


def simulate_reference_schedule(
    conditions,
    electric_power_kw,
    initial_indoor_temperature_c=20.0,
    initial_fabric_temperature_c=20.0,
    measurement_noise_std_c=0.05,
    process_noise_std_c=0.01,
    random_state=42,
    plant_class=TwoStateReferencePlant,
):
    """Apply a fixed schedule to a replaceable reference plant."""

    power = np.asarray(electric_power_kw, dtype=float)
    if len(power) != len(conditions):
        raise ValueError("Power schedule and conditions must have identical lengths.")
    plant = plant_class(
        step_hours=0.5,
        measurement_noise_std_c=measurement_noise_std_c,
        process_noise_std_c=process_noise_std_c,
        random_state=random_state,
    )
    measurement = plant.reset(initial_indoor_temperature_c, initial_fabric_temperature_c)
    records = []
    maximum_power = plant.parameters.heat_pump_max_electric_kw
    for position, (timestamp, row) in enumerate(conditions.iterrows()):
        result = plant.step(
            power[position] / maximum_power,
            row["outdoor_temperature_c"],
            row["solar_irradiance_w_m2"],
            row["internal_gains_kw"],
        )
        records.append(
            {
                "timestamp": timestamp,
                "indoor_temperature_c": measurement,
                "next_indoor_temperature_c": result["indoor_temperature_measured_c"],
                "fabric_temperature_c": result["fabric_temperature_c"],
                "heat_pump_electric_kw": result["heat_pump_electric_kw"],
                "delivered_heat_kw": result["delivered_heat_kw"],
                "cop": result["cop"],
            }
        )
        measurement = result["indoor_temperature_measured_c"]
    return pd.DataFrame.from_records(records).set_index("timestamp")


def simulate_thermostat(
    conditions,
    lower_deadband_c=0.2,
    upper_deadband_c=0.2,
    initial_temperature_c=20.0,
    random_state=42,
    plant_class=TwoStateReferencePlant,
):
    """Simulate a simple on/off thermostat on the reference plant."""

    plant = plant_class(step_hours=0.5, random_state=random_state)
    measurement = plant.reset(initial_temperature_c, initial_temperature_c)
    on = False
    records = []
    for timestamp, row in conditions.iterrows():
        lower = row["comfort_min_c"]
        if measurement < lower - lower_deadband_c:
            on = True
        elif measurement > lower + upper_deadband_c:
            on = False
        result = plant.step(
            float(on),
            row["outdoor_temperature_c"],
            row["solar_irradiance_w_m2"],
            row["internal_gains_kw"],
        )
        records.append(
            {
                "timestamp": timestamp,
                "indoor_temperature_c": measurement,
                "next_indoor_temperature_c": result["indoor_temperature_measured_c"],
                "heat_pump_electric_kw": result["heat_pump_electric_kw"],
            }
        )
        measurement = result["indoor_temperature_measured_c"]
    return pd.DataFrame.from_records(records).set_index("timestamp")


def realised_metrics(results, conditions):
    """Return energy, cost, peak and realised lower-comfort metrics."""

    dt = 0.5
    power = results["heat_pump_electric_kw"].to_numpy()
    temperature = results["next_indoor_temperature_c"].to_numpy()
    lower = conditions["comfort_min_c"].to_numpy()
    violation = np.maximum(lower - temperature, 0.0)
    return {
        "Energy [kWh]": float(power.sum() * dt),
        "Cost [£]": float(np.sum(power * conditions["import_price_gbp_per_kwh"].to_numpy()) * dt),
        "Peak [kW]": float(power.max()),
        "Violation [h]": float(np.sum(violation > 0) * dt),
        "Degree-hours [°C h]": float(violation.sum() * dt),
        "Maximum deviation [°C]": float(violation.max()),
    }

