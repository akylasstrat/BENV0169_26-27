"""Supplied Day 2 restarting cell. Run from the repository root.
Rebuild the fixed point model and aligned handover from source; no prior notebook state.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
ROOT = Path.cwd()
EXPORT_ARTEFACTS = True

PV_PATH = ROOT / "data" / "tutorial_4_2" / "gefcom2014_solar_zone3.csv"
pv_source = pd.read_csv(PV_PATH, index_col="timestamp_utc")
pv_source.index = pd.to_datetime(pv_source.index, utc=True)
pv_source = pv_source.sort_index()
pv_source["forecast_issue_time_utc"] = pd.to_datetime(pv_source["forecast_issue_time_utc"], utc=True)
assert pv_source.index.is_unique and pv_source.zone_id.eq(3).all()
expected_pv_index = pd.date_range(pv_source.index.min(), pv_source.index.max(), freq="h")
assert pv_source.index.equals(expected_pv_index), "Investigate genuine missing PV timestamps."
WEATHER_COLUMNS = ["forecast_surface_solar_radiation_w_m2", "forecast_air_temperature_c",
                   "forecast_total_cloud_cover_fraction"]
# Explicit mapping: these descriptive source names are already in the repository CSV.
pv = pv_source[["solar_power_pu", *WEATHER_COLUMNS,
                "forecast_issue_time_utc", "forecast_lead_hours"]].copy()
pv["solar_power_pu_raw"] = pv["solar_power_pu"].copy()
pv["solar_power_pu"] = pv["solar_power_pu_raw"].clip(lower=0, upper=1)
changed_targets = pv.loc[pv.solar_power_pu != pv.solar_power_pu_raw,
                         ["solar_power_pu_raw", "solar_power_pu"]]
pv["origin_time_utc"] = pv.index - pd.Timedelta(hours=1)
pv["solar_available"] = pv[WEATHER_COLUMNS[0]] > 1.0
assert pv.forecast_issue_time_utc.le(pv.origin_time_utc).all()
assert np.isfinite(pv[["solar_power_pu", *WEATHER_COLUMNS]]).all().all()
PV_TRAIN_START, PV_TRAIN_END = "2012-04-01 01:00+00:00", "2013-03-31 23:00+00:00"
PV_CAL_START, PV_CAL_END = "2013-04-01 00:00+00:00", "2013-06-30 23:00+00:00"
PV_TEST_START, PV_TEST_END = "2013-07-01 00:00+00:00", "2013-09-30 23:00+00:00"
train_index = pv.loc[pd.Timestamp(PV_TRAIN_START):pd.Timestamp(PV_TRAIN_END)].index
calibration_index = pv.loc[pd.Timestamp(PV_CAL_START):pd.Timestamp(PV_CAL_END)].index
test_index = pv.loc[pd.Timestamp(PV_TEST_START):pd.Timestamp(PV_TEST_END)].index
assert [len(train_index), len(calibration_index), len(test_index)] == [8759, 2184, 2208]

pv_features = pv[WEATHER_COLUMNS].copy()
pv_features.insert(0, "pv_lag_1", pv["solar_power_pu"].shift(1))

PV_FEATURE_COLUMNS = ["pv_lag_1", *WEATHER_COLUMNS]
X_pv_train = pv_features.loc[train_index].dropna()
X_pv_calibration = pv_features.loc[calibration_index]
X_pv_test = pv_features.loc[test_index]
y_pv_train = pv.loc[X_pv_train.index, "solar_power_pu"]
y_pv_calibration = pv.loc[calibration_index, "solar_power_pu"]
y_pv_test = pv.loc[test_index, "solar_power_pu"]
assert len(X_pv_train) == 8758  # Only the first target lacks pre-extract lag history.
assert X_pv_calibration.notna().all().all() and X_pv_test.notna().all().all()
assert X_pv_train.index.equals(y_pv_train.index)
assert X_pv_calibration.index.equals(y_pv_calibration.index) and X_pv_test.index.equals(y_pv_test.index)

pv_linear_model = LinearRegression()
pv_linear_model.fit(X_pv_train[PV_FEATURE_COLUMNS], y_pv_train)
fitted_coefficients = pv_linear_model.coef_.copy()
linear_calibration_raw = pd.Series(pv_linear_model.predict(X_pv_calibration[PV_FEATURE_COLUMNS]), index=calibration_index)
linear_test_raw = pd.Series(pv_linear_model.predict(X_pv_test[PV_FEATURE_COLUMNS]), index=test_index)
np.testing.assert_array_equal(fitted_coefficients, pv_linear_model.coef_)

linear_calibration = linear_calibration_raw.clip(lower=0, upper=1)
linear_test = linear_test_raw.clip(lower=0, upper=1)
linear_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
linear_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0

if EXPORT_ARTEFACTS:
    HANDOVER_DIR = ROOT / "data" / "week4_day1" / "day2_handover"
    HANDOVER_DIR.mkdir(parents=True, exist_ok=True)
    for name, index, raw_prediction, prediction in [
        ("calibration", calibration_index, linear_calibration_raw, linear_calibration),
        ("test", test_index, linear_test_raw, linear_test)]:
        bundle = pd.DataFrame(index=index)
        bundle.index.name = "target_time_utc"
        bundle["origin_time_utc"] = pv.loc[index, "origin_time_utc"]
        bundle["weather_issue_time_utc"] = pv.loc[index, "forecast_issue_time_utc"]
        bundle["solar_available"] = pv.loc[index, "solar_available"]
        bundle["actual"] = pv.loc[index, "solar_power_pu"]
        bundle["raw_forecast"] = raw_prediction
        bundle["corrected_forecast"] = prediction
        assert bundle.index.equals(index) and bundle.notna().all().all()
        bundle.to_csv(HANDOVER_DIR / f"{name}_predictions.csv")
    metadata = {
        "schema_version": 1, "frozen_model": "LinearRegression", "configuration": pv_linear_model.get_params(),
        "feature_columns": PV_FEATURE_COLUMNS, "seed": 42,
        "seed_use": "LinearRegression is deterministic; 42 fixes the optional forest, not model selection.",
        "source": {"path": PV_PATH.relative_to(ROOT).as_posix(),
                   "sha256": hashlib.sha256(PV_PATH.read_bytes()).hexdigest(), "zone": 3,
                   "citation": "Hong et al. (2016), doi:10.1016/j.ijforecast.2016.02.001",
                   "clock": "UTC, as interpreted in the source extract README; exact site undisclosed"},
        "split_dates": {"training": [PV_TRAIN_START, PV_TRAIN_END],
                        "calibration": [PV_CAL_START, PV_CAL_END], "test": [PV_TEST_START, PV_TEST_END]},
        "counts": {"training_pre_lag": len(train_index), "training_fit": len(y_pv_train),
                   "calibration": len(calibration_index), "test": len(test_index)},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit_learn": sklearn.__version__},
        "availability": "Origin=target-1h; weather issue<=origin. Assume forecasts are available immediately after issue; publication delays are not recorded.",
        "target_contract": "Teaching target and lag: clip source solar_power_pu to [0,1]; retain raw values separately.",
        "altered_source_targets": {"count": len(changed_targets), "timestamps": changed_targets.index.astype(str).tolist(),
                                    "raw_values": changed_targets.solar_power_pu_raw.tolist(),
                                    "teaching_values": changed_targets.solar_power_pu.tolist()},
        "correction_rules": {"lower": 0.0, "upper": 1.0, "solar_available": "forecast_surface_solar_radiation_w_m2 > 1.0",
                             "zero_rule": "Set both persistence and regression to zero when forecast irradiance is at most 1 W/m2; actual PV may still be positive."},
        "calibration_residual_definition": "actual - corrected_forecast (computed in Day 2, not here)",
        "fitted_coefficients": pv_linear_model.coef_.tolist(), "fitted_intercept": float(pv_linear_model.intercept_),
        "selection": "Linear regression fixed for Day 2 before examining final test scores.",
        "artefact_sha256": {f"{name}_predictions.csv": hashlib.sha256((HANDOVER_DIR / f"{name}_predictions.csv").read_bytes()).hexdigest()
                            for name in ["calibration", "test"]},
        "release": "Release the starting bundle and standalone rebuilding cell on Day 2. No assessment data/answers are included.",
    }
    (HANDOVER_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved aligned calibration/test predictions and explicit provenance; no pickle required.")
