"""Build the compact GEFCom2014 solar dataset used in Tutorial 4.2.

The source CSV contains three Australian PV zones, normalised power and
24-hour numerical-weather-prediction trajectories. It is never modified.

Example
-------
python scripts/build_tutorial_4_2_pv_dataset.py --source-csv "PATH/gefcom2014-solar.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ZONE_ID = 3
REQUIRED_COLUMNS = [
    "TIMESTAMP", "ZONEID", "VAR78", "VAR79", "VAR134", "VAR157",
    "VAR164", "VAR165", "VAR166", "VAR167", "VAR169", "VAR175",
    "VAR178", "VAR228", "POWER",
]


def _decumulate(
    data: pd.DataFrame,
    source_column: str,
    issue_column: str = "forecast_issue_time_utc",
) -> tuple[pd.Series, int]:
    """Convert a within-run accumulation to one-hour increments."""
    increments = data.groupby(issue_column, sort=False)[source_column].diff()
    first_in_run = data.groupby(issue_column, sort=False).cumcount().eq(0)
    increments.loc[first_in_run] = data.loc[first_in_run, source_column]
    negative_count = int(increments.lt(0).sum())
    return increments.clip(lower=0), negative_count


def build_dataset(source_csv: Path, output_path: Path) -> pd.DataFrame:
    """Construct and validate the Zone 3 teaching extract."""
    source = pd.read_csv(source_csv)
    missing_columns = sorted(set(REQUIRED_COLUMNS).difference(source.columns))
    if missing_columns:
        raise ValueError(f"Source CSV is missing columns: {missing_columns}")

    source["TIMESTAMP"] = pd.to_datetime(source["TIMESTAMP"], utc=True)
    data = (
        source.loc[source["ZONEID"].eq(ZONE_ID), REQUIRED_COLUMNS]
        .sort_values("TIMESTAMP")
        .reset_index(drop=True)
    )
    if data.empty:
        raise ValueError(f"No observations found for zone {ZONE_ID}.")
    if data["TIMESTAMP"].duplicated().any():
        raise ValueError("Duplicate Zone 3 timestamps found in the source CSV.")

    expected_index = pd.date_range(
        data["TIMESTAMP"].min(), data["TIMESTAMP"].max(), freq="h", tz="UTC"
    )
    observed_index = pd.DatetimeIndex(data["TIMESTAMP"])
    if not observed_index.equals(expected_index):
        missing = expected_index.difference(observed_index)
        raise ValueError(
            f"Zone 3 is not a complete hourly series; {len(missing)} timestamps are missing."
        )
    if data[REQUIRED_COLUMNS[2:]].isna().any().any():
        raise ValueError("Unexpected missing values found in Zone 3.")

    # Forecasts are issued at 00:00 UTC. Valid times 01:00, ..., 00:00 on
    # the following date correspond to lead hours 1, ..., 24.
    data["forecast_issue_time_utc"] = (
        data["TIMESTAMP"] - pd.Timedelta(hours=1)
    ).dt.floor("D")
    data["forecast_lead_hours"] = (
        (data["TIMESTAMP"] - data["forecast_issue_time_utc"])
        .dt.total_seconds()
        .div(3600)
        .astype(int)
    )
    if set(data["forecast_lead_hours"].unique()) != set(range(1, 25)):
        raise ValueError("Expected complete 24-hour trajectories with leads 1 to 24.")

    ssrd_increment, ssrd_negatives = _decumulate(data, "VAR169")
    strd_increment, strd_negatives = _decumulate(data, "VAR175")
    tsr_increment, tsr_negatives = _decumulate(data, "VAR178")
    precipitation_increment, precipitation_negatives = _decumulate(data, "VAR228")

    output = pd.DataFrame(index=observed_index)
    output.index.name = "timestamp_utc"
    output["zone_id"] = ZONE_ID
    output["forecast_issue_time_utc"] = data["forecast_issue_time_utc"].to_numpy()
    output["forecast_lead_hours"] = data["forecast_lead_hours"].to_numpy()
    output["solar_power_pu"] = data["POWER"].to_numpy()

    output["forecast_total_column_cloud_liquid_water_kg_m2"] = data["VAR78"].to_numpy()
    output["forecast_total_column_cloud_ice_water_kg_m2"] = data["VAR79"].to_numpy()
    output["forecast_surface_pressure_pa"] = data["VAR134"].to_numpy()
    output["forecast_relative_humidity_pct"] = data["VAR157"].to_numpy()
    output["forecast_total_cloud_cover_fraction"] = data["VAR164"].to_numpy()
    output["forecast_wind_u_10m_m_s"] = data["VAR165"].to_numpy()
    output["forecast_wind_v_10m_m_s"] = data["VAR166"].to_numpy()
    output["forecast_wind_speed_10m_m_s"] = np.hypot(
        data["VAR165"].to_numpy(), data["VAR166"].to_numpy()
    )
    output["forecast_air_temperature_c"] = data["VAR167"].to_numpy() - 273.15

    output["forecast_surface_solar_radiation_accumulated_j_m2"] = data["VAR169"].to_numpy()
    output["forecast_surface_solar_radiation_w_m2"] = ssrd_increment.to_numpy() / 3600
    output["forecast_surface_thermal_radiation_accumulated_j_m2"] = data["VAR175"].to_numpy()
    output["forecast_surface_thermal_radiation_w_m2"] = strd_increment.to_numpy() / 3600
    output["forecast_top_net_solar_radiation_accumulated_j_m2"] = data["VAR178"].to_numpy()
    output["forecast_top_net_solar_radiation_w_m2"] = tsr_increment.to_numpy() / 3600
    output["forecast_total_precipitation_accumulated_m"] = data["VAR228"].to_numpy()
    output["forecast_precipitation_mm"] = precipitation_increment.to_numpy() * 1000

    # Exact site coordinates are undisclosed. Forecast irradiance therefore
    # supplies a reproducible daylight flag without using realised PV output.
    output["is_daylight_forecast"] = (
        output["forecast_surface_solar_radiation_w_m2"] > 1.0
    ).astype(int)

    if (output["solar_power_pu"] < 0).any():
        raise ValueError("Solar power contains negative values.")
    if output.isna().any().any():
        missing = output.isna().sum()
        raise ValueError(f"Unexpected missing values found:\n{missing[missing.gt(0)]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, float_format="%.6f")

    print("Clipped negative one-hour increments (numerical precision):")
    print({
        "VAR169": ssrd_negatives,
        "VAR175": strd_negatives,
        "VAR178": tsr_negatives,
        "VAR228": precipitation_negatives,
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-csv",
        type=Path,
        required=True,
        help="Path to the downloaded gefcom2014-solar.csv file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tutorial_4_2/gefcom2014_solar_zone3.csv"),
        help="Destination CSV in the repository.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    dataset = build_dataset(arguments.source_csv, arguments.output)
    print(f"Wrote {len(dataset):,} rows to {arguments.output}")
    print(dataset[["solar_power_pu", "forecast_surface_solar_radiation_w_m2"]].describe())
