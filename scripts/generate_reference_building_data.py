"""Generate deterministic teaching data for the Weeks 6--8 tutorials."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from course_utils.reference_building import (
    ReferenceBuildingParameters,
    TwoStateReferencePlant,
)


DATA_DIR = ROOT / "data" / "reference_building"
STEP_HOURS = 0.5


def make_conditions(index, random_state):
    rng = np.random.default_rng(random_state)
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    day_number = np.arange(len(index)) * STEP_HOURS / 24

    synoptic = 2.2 * np.sin(2 * np.pi * day_number / 6.5)
    daily = 2.8 * np.sin(2 * np.pi * (hour - 14) / 24)
    outdoor = 5.0 + synoptic + daily + rng.normal(0, 0.35, len(index))

    daylight = np.sin(np.pi * np.clip((hour - 8.0) / 8.0, 0, 1))
    cloud = np.clip(0.65 + 0.22 * np.sin(2 * np.pi * day_number / 3.7) + rng.normal(0, 0.12, len(index)), 0.15, 1.0)
    irradiance = np.maximum(0.0, 320.0 * daylight * cloud)

    weekday = index.dayofweek.to_numpy() < 5
    occupied = np.where(weekday, (hour < 7) | (hour >= 20), True).astype(int)
    internal_gains = 0.25 + 0.75 * occupied
    comfort_min = np.where(occupied, 20.0, 17.0)
    comfort_max = np.where(occupied, 23.0, 25.0)

    base_load = 0.35 + 0.25 * occupied + 0.08 * np.sin(2 * np.pi * hour / 24) ** 2
    pv_available = np.minimum(4.0, 4.0 * irradiance / 650.0)
    import_price = np.where(hour < 7, 0.18, np.where((hour >= 16) & (hour < 19), 0.42, 0.29))

    return pd.DataFrame(
        {
            "outdoor_temperature_c": outdoor,
            "solar_irradiance_w_m2": irradiance,
            "internal_gains_kw": internal_gains,
            "occupancy": occupied,
            "comfort_min_c": comfort_min,
            "comfort_max_c": comfort_max,
            "base_load_kw": base_load,
            "pv_available_kw": pv_available,
            "import_price_gbp_per_kwh": import_price,
            "export_price_gbp_per_kwh": 0.08,
        },
        index=index,
    )


def generate_identification_data():
    index = pd.date_range("2025-01-06", periods=28 * 48, freq="30min", name="timestamp")
    conditions = make_conditions(index, random_state=101)
    plant = TwoStateReferencePlant(
        step_hours=STEP_HOURS,
        measurement_noise_std_c=0.06,
        process_noise_std_c=0.012,
        random_state=202,
    )
    measurement = plant.reset(20.0, 20.0)
    rng = np.random.default_rng(303)
    block_levels = rng.choice([0.0, 0.18, 0.35, 0.55, 0.72], size=int(np.ceil(len(index) / 4)))

    records = []
    for position, (timestamp, row) in enumerate(conditions.iterrows()):
        modulation = float(block_levels[position // 4])
        if measurement < 18.2:
            modulation = max(modulation, 0.55)
        if measurement > 23.0:
            modulation = 0.0

        electric_kw, delivered_kw, cop = plant.actuation(modulation, row["outdoor_temperature_c"])
        records.append(
            {
                "timestamp": timestamp,
                "indoor_temperature_c": measurement,
                "outdoor_temperature_c": row["outdoor_temperature_c"],
                "heat_pump_modulation": modulation,
                "heat_pump_electric_kw": electric_kw,
                "delivered_heat_kw": delivered_kw,
                "cop": cop,
                "solar_irradiance_w_m2": row["solar_irradiance_w_m2"],
                "internal_gains_kw": row["internal_gains_kw"],
                "occupancy": row["occupancy"],
            }
        )
        result = plant.step(
            modulation,
            row["outdoor_temperature_c"],
            row["solar_irradiance_w_m2"],
            row["internal_gains_kw"],
        )
        measurement = result["indoor_temperature_measured_c"]

    return pd.DataFrame.from_records(records).set_index("timestamp")


def generate_forecasts(conditions, horizon_steps=48):
    rng = np.random.default_rng(515)
    records = []
    for origin_position, issue_time in enumerate(conditions.index[:-1]):
        final_position = min(origin_position + horizon_steps, len(conditions) - 1)
        for valid_position in range(origin_position + 1, final_position + 1):
            valid_time = conditions.index[valid_position]
            lead_steps = valid_position - origin_position
            actual = conditions.iloc[valid_position]
            temperature_error = 0.20 + rng.normal(0, 0.18 + 0.008 * lead_steps)
            solar_factor = max(0.0, 1.0 + rng.normal(0, 0.08 + 0.003 * lead_steps))
            solar_forecast = max(0.0, actual["solar_irradiance_w_m2"] * solar_factor)
            records.append(
                {
                    "issue_time": issue_time,
                    "valid_time": valid_time,
                    "lead_steps": lead_steps,
                    "outdoor_temperature_forecast_c": actual["outdoor_temperature_c"] + temperature_error,
                    "solar_irradiance_forecast_w_m2": solar_forecast,
                    "pv_available_forecast_kw": min(4.0, 4.0 * solar_forecast / 650.0),
                }
            )
    return pd.DataFrame.from_records(records)


def fit_verified_controller_model(identification_data):
    from scipy.optimize import lsq_linear

    data = identification_data.copy()
    data["temperature_change_c"] = data["indoor_temperature_c"].shift(-1) - data["indoor_temperature_c"]
    data = data.iloc[:-1]
    design = np.column_stack(
        [
            data["outdoor_temperature_c"] - data["indoor_temperature_c"],
            data["delivered_heat_kw"],
        ]
    )
    target = data["temperature_change_c"].to_numpy()
    fit = lsq_linear(design, target, bounds=([1e-5, 1e-5], [0.4, 0.5]))
    loss_coefficient, heat_coefficient = fit.x
    capacitance = STEP_HOURS / heat_coefficient
    resistance = heat_coefficient / loss_coefficient
    return {
        "sampling_interval_hours": STEP_HOURS,
        "loss_coefficient_per_step": float(loss_coefficient),
        "heat_coefficient_c_per_kw_step": float(heat_coefficient),
        "effective_resistance_k_per_kw": float(resistance),
        "effective_capacitance_kwh_per_k": float(capacitance),
        "effective_time_constant_hours": float(resistance * capacitance),
        "nominal_cop": 3.0,
        "heat_pump_max_electric_kw": 5.0,
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    identification = generate_identification_data()
    operation_index = pd.date_range("2025-02-03", periods=14 * 48, freq="30min", name="timestamp")
    operation = make_conditions(operation_index, random_state=404)
    forecasts = generate_forecasts(operation)
    controller_model = fit_verified_controller_model(identification)

    identification.to_csv(DATA_DIR / "identification_data.csv", float_format="%.5f")
    operation.to_csv(DATA_DIR / "operation_conditions.csv", float_format="%.5f")
    forecasts.to_csv(DATA_DIR / "operation_forecasts.csv", index=False, float_format="%.5f")
    (DATA_DIR / "verified_controller_model.json").write_text(
        json.dumps(controller_model, indent=2), encoding="utf-8"
    )
    parameters = ReferenceBuildingParameters().to_dict()
    (DATA_DIR / "reference_plant_parameters.json").write_text(
        json.dumps(parameters, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
