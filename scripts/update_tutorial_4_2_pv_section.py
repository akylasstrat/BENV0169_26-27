"""Replace the provisional PV section in both Tutorial 4.2 notebooks.

The script preserves all preceding user-edited cells and changes only the
shared introduction/import plus the former placeholder section.
"""

from __future__ import annotations

import json
from pathlib import Path


STUDENT_PATH = Path("tutorial_4_2_pv_probabilistic_forecasting.ipynb")
SOLUTION_PATH = Path("tutorial_4_2_pv_probabilistic_forecasting_solution.ipynb")


INTRODUCTION = """# Tutorial 4.2: PV and probabilistic forecasting

This tutorial extends the day-ahead forecasting workflow from Tutorial 4.1. We first use the GEFCom2012 aggregate load to study forecast uncertainty, because its data and point-forecast specification are already familiar. We then apply the same principles to normalised PV output and day-ahead ECMWF weather forecasts from the GEFCom2014 solar track.

The main distinction is between:

- a **point forecast**, which provides one value at each forecast horizon;
- a **prediction interval**, which provides marginal lower and upper bounds; and
- a **scenario forecast**, which provides complete trajectories and preserves relationships between hours.

## Learning outcomes

By the end of the tutorial, you should be able to:

1. construct constant and hour-dependent residual prediction intervals;
2. fit linear quantile-regression models;
3. assess coverage, sharpness, quantile score and interval score;
4. generate independent and temporally correlated Gaussian scenarios; and
5. apply physical constraints to PV point forecasts and prediction intervals.

## Indicative schedule

- set-up, point forecast and residual diagnosis: 10 minutes;
- residual-based intervals: 15 minutes;
- quantile regression and interval evaluation: 15 minutes;
- Gaussian trajectory scenarios: 15 minutes;
- PV point forecasting and bad practices: 25 minutes;
- PV intervals, quantiles and scenarios: 20 minutes;
- discussion: 5 minutes.
"""


LOAD_QUANTILE_MARKDOWN = r"""## 5. Task 3 — Linear quantile regression

Residual intervals shift one point forecast using errors observed on the calibration set. Quantile regression instead estimates a conditional quantile directly. We fit separate linear models for

$$
q \in \{0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95\}
$$

using pinball loss. These quantiles define central 50%, 70%, 80% and 90% prediction intervals. The additional intervals let us assess whether empirical coverage tracks nominal coverage across several probability levels, rather than judging calibration from one 90% interval alone.

The models use the training period only. The calibration period remains reserved for the residual methods, which keeps the comparison transparent. Separately fitted quantiles can cross; the code counts any adjacent crossings and sorts the forecasts at each timestamp before constructing intervals.
"""


LOAD_QUANTILE_SOLUTION = """QUANTILES = [
    0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95
]
quantile_models = {}
raw_quantile_predictions = pd.DataFrame(index=test.index)

for quantile in QUANTILES:
    model = make_pipeline(
        StandardScaler(),
        QuantileRegressor(quantile=quantile, alpha=0.01, solver="highs"),
    )
    model.fit(train[FEATURE_COLUMNS], train["target"])
    quantile_models[quantile] = model
    raw_quantile_predictions[quantile] = model.predict(test[FEATURE_COLUMNS])

crossing_mask = (
    raw_quantile_predictions.diff(axis=1).iloc[:, 1:].lt(0).any(axis=1)
)
print("Timestamps with crossing quantiles:", crossing_mask.sum())

# Sorting each row gives non-decreasing reported quantiles. This simple repair
# is transparent, although joint/non-crossing quantile models are preferable
# in applications where the complete distribution is the main output.
ordered_predictions = np.sort(raw_quantile_predictions.to_numpy(), axis=1)
quantile_predictions = pd.DataFrame(
    ordered_predictions,
    index=test.index,
    columns=QUANTILES,
)

intervals["Linear quantile regression"] = (
    quantile_predictions[0.05].rename("Lower"),
    quantile_predictions[0.95].rename("Upper"),
)
"""


LOAD_QUANTILE_STUDENT = """QUANTILES = [
    0.05, 0.10, 0.15, 0.25, 0.50, 0.75, 0.85, 0.90, 0.95
]
quantile_models = {}
raw_quantile_predictions = pd.DataFrame(index=test.index)

for quantile in QUANTILES:
    # TODO 3.1: create a StandardScaler + QuantileRegressor pipeline.
    # Use alpha=0.01 and solver="highs". Fit it on train, then predict test.
    pass

if set(quantile_models) != set(QUANTILES):
    raise NotImplementedError("Fit all nine quantile-regression models.")

# TODO 3.2: identify rows where any quantile exceeds the next quantile.
crossing_mask = ...
if crossing_mask is ...:
    raise NotImplementedError("Identify quantile crossings.")
print("Timestamps with crossing quantiles:", crossing_mask.sum())

# Sort each row so the reported quantiles are ordered.
ordered_predictions = np.sort(raw_quantile_predictions.to_numpy(), axis=1)
quantile_predictions = pd.DataFrame(
    ordered_predictions,
    index=test.index,
    columns=QUANTILES,
)

intervals["Linear quantile regression"] = (
    quantile_predictions[0.05].rename("Lower"),
    quantile_predictions[0.95].rename("Upper"),
)
"""


LOAD_COVERAGE_MARKDOWN = """### 5.1 Coverage-calibration curve

A coverage-calibration curve is analogous to a reliability diagram for prediction intervals. Each point compares a central interval's nominal coverage with the proportion of test observations that actually fall between its bounds.

- A point on the diagonal is calibrated at that coverage level.
- A point below the diagonal indicates **undercoverage**: the interval covers fewer observations than promised and may be too narrow or biased.
- A point above the diagonal indicates **overcoverage**: the interval covers more observations than promised and may be unnecessarily wide.

This is not a classical Q–Q plot of two distributions, but its diagonal interpretation is similar and is often more direct for explaining interval calibration.

The plot also includes two deliberately exaggerated **synthetic** points so that undercoverage and overcoverage remain visually obvious on a lecture slide. Cross markers identify these teaching examples; circular markers are the results calculated from GEFCom2012.
"""


LOAD_COVERAGE_CODE = """CENTRAL_INTERVAL_QUANTILES = {
    "50% interval": (0.25, 0.75),
    "70% interval": (0.15, 0.85),
    "80% interval": (0.10, 0.90),
    "90% interval": (0.05, 0.95),
}

coverage_rows = []
for interval_name, (lower_quantile, upper_quantile) in CENTRAL_INTERVAL_QUANTILES.items():
    nominal_coverage = 100 * (upper_quantile - lower_quantile)
    observed_coverage = empirical_coverage(
        test["target"],
        quantile_predictions[lower_quantile],
        quantile_predictions[upper_quantile],
    )
    coverage_rows.append({
        "Interval": interval_name,
        "Nominal coverage (%)": nominal_coverage,
        "Empirical coverage (%)": observed_coverage,
        "Coverage gap (percentage points)": observed_coverage - nominal_coverage,
    })

coverage_calibration = pd.DataFrame(coverage_rows).set_index("Interval")
display(coverage_calibration.round(1))

# Deliberately exaggerated teaching examples. These are not model results.
synthetic_coverage_examples = pd.DataFrame(
    {
        "Nominal coverage (%)": [60.0, 85.0],
        "Empirical coverage (%)": [82.0, 60.0],
    },
    index=["Synthetic overcoverage", "Synthetic undercoverage"],
)
synthetic_coverage_examples["Coverage gap (percentage points)"] = (
    synthetic_coverage_examples["Empirical coverage (%)"]
    - synthetic_coverage_examples["Nominal coverage (%)"]
)

fig, ax = plt.subplots(figsize=(7, 7))
diagonal = np.linspace(0, 100, 201)
ax.fill_between(
    diagonal, 0, diagonal,
    color="#D55E00", alpha=0.08, label="Undercoverage",
)
ax.fill_between(
    diagonal, diagonal, 100,
    color="#0072B2", alpha=0.08, label="Overcoverage",
)
ax.plot(diagonal, diagonal, color="0.25", linewidth=1.5, label="Perfect calibration")

for interval_name, row in coverage_calibration.iterrows():
    gap = row["Coverage gap (percentage points)"]
    point_colour = "#D55E00" if gap < 0 else "#0072B2"
    ax.plot(
        [row["Nominal coverage (%)"], row["Nominal coverage (%)"]],
        [row["Nominal coverage (%)"], row["Empirical coverage (%)"]],
        color=point_colour,
        linewidth=2,
        zorder=2,
    )
    ax.scatter(
        row["Nominal coverage (%)"],
        row["Empirical coverage (%)"],
        s=90,
        color=point_colour,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
    )
    ax.annotate(
        f"{interval_name} ({gap:+.1f} pp)",
        (row["Nominal coverage (%)"], row["Empirical coverage (%)"]),
        xytext=(7, -10 if gap < 0 else 7),
        textcoords="offset points",
    )

for example_name, row in synthetic_coverage_examples.iterrows():
    gap = row["Coverage gap (percentage points)"]
    point_colour = "#0072B2" if gap > 0 else "#D55E00"
    ax.plot(
        [row["Nominal coverage (%)"], row["Nominal coverage (%)"]],
        [row["Nominal coverage (%)"], row["Empirical coverage (%)"]],
        color=point_colour,
        linewidth=2.5,
        linestyle="--",
        zorder=2,
    )
    ax.scatter(
        row["Nominal coverage (%)"],
        row["Empirical coverage (%)"],
        s=170,
        marker="X",
        color=point_colour,
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
    )
    label_offset = (8, 8) if gap > 0 else (-155, -15)
    ax.annotate(
        f"{example_name} ({gap:+.0f} pp)",
        (row["Nominal coverage (%)"], row["Empirical coverage (%)"]),
        xytext=label_offset,
        textcoords="offset points",
        fontweight="bold",
    )

# Empty artists provide an unambiguous marker key without duplicating labels.
ax.scatter(
    [], [], s=90, marker="o", color="0.45", edgecolor="white",
    label="GEFCom2012 intervals",
)
ax.scatter(
    [], [], s=140, marker="X", color="0.65", edgecolor="black",
    label="Synthetic teaching examples",
)

ax.set(
    xlim=(45, 95),
    ylim=(45, 95),
    aspect="equal",
    xlabel="Nominal interval coverage (%)",
    ylabel="Empirical test coverage (%)",
    title="Coverage calibration of linear quantile-regression intervals\\n"
          "(vertical segments show the coverage gap)",
)
ax.set_xticks(range(50, 91, 10))
ax.set_yticks(range(50, 91, 10))
ax.legend(loc="lower left")
plt.tight_layout()
plt.show()
"""


PV_OVERVIEW = r"""## 7. PV point and probabilistic forecasting

We now use `gefcom2014_solar_zone3.csv`, a cleaned extract from the GEFCom2014 solar track. The competition data contain normalised PV power for three Australian zones and 24-hour ECMWF weather-forecast trajectories issued at 00:00 UTC. We use Zone 3 because its PV output has the strongest contemporaneous relationship with the derived surface-solar-radiation forecast. The precise plant locations are not disclosed. The source timestamps have no explicit offset; we interpret them as UTC because daylight spans the evening and early-morning UTC hours and the mean solar peak occurs near 02:00 UTC, which is consistent with midday in eastern Australia.

The source weather fields are genuine forecasts. The compact extract gives the variables descriptive names while retaining the accumulated radiation and precipitation fields. For accumulated variables, a value at lead $h$ represents the total since the forecast origin. We therefore difference consecutive lead times within each 24-hour trajectory and divide accumulated energy by 3600 seconds to obtain hourly mean flux in $\mathrm{W/m^2}$. At lead 1, the accumulated value is already the increment since the 00:00 forecast origin. Tiny negative differences caused by numerical precision are clipped to zero.

The source identifiers map to ECMWF parameters as follows:

| Source | Forecast variable |
|---|---|
| `VAR78`, `VAR79` | total-column cloud liquid water and ice water |
| `VAR134`, `VAR157`, `VAR164` | surface pressure, relative humidity and total cloud cover |
| `VAR165`, `VAR166`, `VAR167` | 10 m wind components and 2 m air temperature |
| `VAR169`, `VAR175`, `VAR178` | accumulated surface solar, surface thermal and top net solar radiation |
| `VAR228` | accumulated total precipitation |

The cleaned CSV uses descriptive, unit-bearing column names and also provides derived wind speed, hourly radiation fluxes and hourly precipitation. The original anonymous names are documented in `data/tutorial_4_2/README.md`.

The PV data add two physical requirements:

- normalised power should not be negative or exceed 1 p.u.; and
- power must be zero at night.

Because the competition does not publish the site coordinates, `is_daylight_forecast` is based only on forecast surface solar radiation exceeding $1\ \mathrm{W/m^2}$. It does not use realised PV output and therefore avoids leakage. See `data/tutorial_4_2/README.md` for the complete variable mapping and provenance.

MAPE is not used for PV because its denominator is zero at night and can be very small near sunrise and sunset.
"""


PV_LOAD_CODE = """PV_DATA_PATH = Path("data") / "tutorial_4_2" / "gefcom2014_solar_zone3.csv"

pv_data = pd.read_csv(PV_DATA_PATH)
pv_data["timestamp_utc"] = pd.to_datetime(pv_data["timestamp_utc"], utc=True)
pv_data["forecast_issue_time_utc"] = pd.to_datetime(
    pv_data["forecast_issue_time_utc"], utc=True
)
pv_data = pv_data.set_index("timestamp_utc").sort_index()

expected_pv_index = pd.date_range(
    "2012-04-01 01:00", "2014-07-01 00:00", freq="h", tz="UTC"
)

assert pv_data.index.equals(expected_pv_index)
assert pv_data.index.is_unique
assert (pv_data["solar_power_pu"] >= 0).all()
assert set(pv_data["forecast_lead_hours"].unique()) == set(range(1, 25))
assert (pv_data["forecast_issue_time_utc"].dt.hour == 0).all()
assert not pv_data.isna().any().any()

pv_quality_summary = pd.Series({
    "Rows": len(pv_data),
    "Missing timestamps": len(expected_pv_index.difference(pv_data.index)),
    "Duplicate timestamps": pv_data.index.duplicated().sum(),
    "Missing values": int(pv_data.isna().sum().sum()),
    "Minimum normalised PV power (p.u.)": pv_data["solar_power_pu"].min(),
    "Maximum normalised PV power (p.u.)": pv_data["solar_power_pu"].max(),
    "Forecast-daylight hours": int(pv_data["is_daylight_forecast"].sum()),
})
display(pv_quality_summary.to_frame("Value"))
"""


PV_EXPLORATION_CODE = """example_week = pv_data.loc["2013-01-14":"2013-01-20 23:00"]

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(example_week.index, example_week["solar_power_pu"], color="tab:orange")
axes[0].set_ylabel("Normalised PV power (p.u.)")
axes[0].set_title("GEFCom2014 Zone 3: one summer week")

axes[1].plot(
    example_week.index,
    example_week["forecast_surface_solar_radiation_w_m2"],
    color="tab:blue",
    label="Forecast surface solar radiation",
)
axes[1].set_ylabel("Forecast solar radiation (W/m²)")
axes[1].set_xlabel("Valid time (UTC)")
axes[1].legend()
plt.tight_layout()
plt.show()

mean_profiles = pv_data.groupby(pv_data.index.hour)[
    ["solar_power_pu", "forecast_surface_solar_radiation_w_m2"]
].mean()

fig, ax_power = plt.subplots(figsize=(10, 4))
ax_radiation = ax_power.twinx()
ax_power.plot(mean_profiles.index, mean_profiles["solar_power_pu"], color="tab:orange", label="Mean PV power")
ax_radiation.plot(
    mean_profiles.index,
    mean_profiles["forecast_surface_solar_radiation_w_m2"],
    color="tab:blue",
    label="Mean aligned forecast radiation",
)
ax_power.set(xlabel="Hour of day (UTC)", ylabel="Normalised PV power (p.u.)", xticks=range(0, 24, 2))
ax_radiation.set_ylabel("Forecast solar radiation (W/m²)")
lines = ax_power.lines + ax_radiation.lines
ax_power.legend(lines, [line.get_label() for line in lines], loc="upper left")
ax_power.set_title("Mean daily PV and forecast-radiation profiles")
plt.tight_layout()
plt.show()
"""


PV_POINT_MARKDOWN = """### Task 5 — Compare two simple linear PV models

We forecast the full test period directly from target-hour weather forecasts and calendar information. No realised test-period PV values are used as predictors.

Compare:

1. one pooled linear regression with a few explicit interaction terms; and
2. 24 separate linear regressions, one for each UTC hour.

The pooled model shares information across the day. Its interactions allow the radiation slope to change with cloud cover, temperature and time of day. The hourly approach is easy to interpret but fits each model using only one twenty-fourth of the training rows. We select between them on a validation period, not on the final test period.

The chronological roles are:

- April 2012–March 2013: fit the two candidate point models;
- April–June 2013: compare and select the formulation;
- July–September 2013: calibrate prediction intervals; and
- October–December 2013: final test.

After selecting the formulation, we refit it using all data available through June 2013 before generating calibration and test forecasts.
"""


PV_FEATURE_CODE = """def make_pv_features(data):
    features = data.copy()
    hour = features.index.hour
    day_of_year = features.index.dayofyear

    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["day_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    features["day_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    irradiance = features["forecast_surface_solar_radiation_w_m2"]
    features["irradiance_x_cloud"] = (
        irradiance * features["forecast_total_cloud_cover_fraction"]
    )
    features["irradiance_x_temperature"] = (
        irradiance * features["forecast_air_temperature_c"]
    )
    features["irradiance_x_hour_sin"] = irradiance * features["hour_sin"]
    features["irradiance_x_hour_cos"] = irradiance * features["hour_cos"]
    return features


HOURLY_PV_FEATURES = [
    "forecast_surface_solar_radiation_w_m2",
    "forecast_air_temperature_c",
    "forecast_total_cloud_cover_fraction",
    "forecast_relative_humidity_pct",
    "forecast_wind_speed_10m_m_s",
    "forecast_total_column_cloud_liquid_water_kg_m2",
    "day_sin",
    "day_cos",
]

POOLED_PV_FEATURES = HOURLY_PV_FEATURES + [
    "hour_sin",
    "hour_cos",
    "irradiance_x_cloud",
    "irradiance_x_temperature",
    "irradiance_x_hour_sin",
    "irradiance_x_hour_cos",
]

pv_features = make_pv_features(pv_data)
issue_dates = pv_features["forecast_issue_time_utc"]
pv_train = pv_features.loc[issue_dates < "2013-04-01"]
pv_validation = pv_features.loc[
    (issue_dates >= "2013-04-01") & (issue_dates < "2013-07-01")
]
pv_calibration = pv_features.loc[
    (issue_dates >= "2013-07-01") & (issue_dates < "2013-10-01")
]
pv_test = pv_features.loc[
    (issue_dates >= "2013-10-01") & (issue_dates < "2014-01-01")
]

pd.DataFrame({
    "Start": [frame.index.min() for frame in [pv_train, pv_validation, pv_calibration, pv_test]],
    "End": [frame.index.max() for frame in [pv_train, pv_validation, pv_calibration, pv_test]],
    "Hours": [len(frame) for frame in [pv_train, pv_validation, pv_calibration, pv_test]],
}, index=["Training", "Validation", "Calibration", "Test"])
"""


PV_MODEL_SOLUTION = """def fit_hourly_linear_models(training_data, feature_columns):
    models = {}
    for hour in range(24):
        hour_rows = training_data.index.hour == hour
        model = LinearRegression()
        model.fit(
            training_data.loc[hour_rows, feature_columns],
            training_data.loc[hour_rows, "solar_power_pu"],
        )
        models[hour] = model
    return models


def predict_hourly_linear_models(models, forecast_data, feature_columns):
    forecast = pd.Series(index=forecast_data.index, dtype=float, name="PV forecast")
    for hour, model in models.items():
        hour_rows = forecast_data.index.hour == hour
        forecast.loc[hour_rows] = model.predict(
            forecast_data.loc[hour_rows, feature_columns]
        )
    return forecast


pooled_candidate = LinearRegression()
pooled_candidate.fit(pv_train[POOLED_PV_FEATURES], pv_train["solar_power_pu"])
pooled_validation_raw = pd.Series(
    pooled_candidate.predict(pv_validation[POOLED_PV_FEATURES]),
    index=pv_validation.index,
    name="Pooled interactions",
)

hourly_candidates = fit_hourly_linear_models(pv_train, HOURLY_PV_FEATURES)
hourly_validation_raw = predict_hourly_linear_models(
    hourly_candidates, pv_validation, HOURLY_PV_FEATURES
).rename("24 hourly models")
"""


PV_MODEL_STUDENT = """def fit_hourly_linear_models(training_data, feature_columns):
    \"\"\"Fit and return one LinearRegression model for each hour 0, ..., 23.\"\"\"
    # TODO 5.1: select the rows for each hour, fit a model and store it in
    # a dictionary keyed by hour.
    raise NotImplementedError("Fit the 24 hourly PV models.")


def predict_hourly_linear_models(models, forecast_data, feature_columns):
    \"\"\"Return one time-indexed forecast assembled from the 24 models.\"\"\"
    # TODO 5.2: use the model matching each row's UTC hour.
    raise NotImplementedError("Generate forecasts from the 24 hourly models.")


# TODO 5.3: fit one pooled LinearRegression with POOLED_PV_FEATURES and
# predict the validation period.
pooled_candidate = ...
pooled_validation_raw = ...

if pooled_candidate is ... or pooled_validation_raw is ...:
    raise NotImplementedError("Fit and evaluate the pooled interaction model.")

hourly_candidates = fit_hourly_linear_models(pv_train, HOURLY_PV_FEATURES)
hourly_validation_raw = predict_hourly_linear_models(
    hourly_candidates, pv_validation, HOURLY_PV_FEATURES
).rename("24 hourly models")
"""


PV_CONSTRAINT_MARKDOWN = r"""### Task 6 — Apply PV physical constraints

Ordinary linear regression does not know the feasible range of PV power. A raw forecast can therefore be negative or positive at night. These are distinct mistakes: projecting negative values to zero still leaves any positive night-time values untouched.

Apply the constraints in this order:

$$
\hat{p}^{\mathrm{physical}}_t =
\begin{cases}
0, & \text{if the Sun is below the horizon},\\
\min\!\left(p_{\max},\max(0,\hat{p}_t)\right), & \text{during daylight}.
\end{cases}
$$

Because the competition target is normalised, we use 1 p.u. as $p_{\max}$. The source contains a few observations marginally above 1 because of normalisation and rounding; these are retained. Compare the candidates after this common post-processing step, select the lower validation RMSE, and only then evaluate the chosen formulation on the untouched test period.
"""


PV_CONSTRAINT_SOLUTION = """def apply_pv_constraints(point_forecast, forecast_data):
    corrected = point_forecast.copy()
    corrected = corrected.clip(lower=0, upper=1.0)
    corrected.loc[forecast_data["is_daylight_forecast"].eq(0)] = 0
    return corrected


def rmse(actual, forecast):
    return np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2))


def mae(actual, forecast):
    return np.mean(np.abs(np.asarray(actual) - np.asarray(forecast)))


raw_validation_forecasts = {
    "Pooled interactions": pooled_validation_raw,
    "24 hourly models": hourly_validation_raw,
}

validation_rows = {}
for model_name, raw_forecast in raw_validation_forecasts.items():
    corrected_forecast = apply_pv_constraints(raw_forecast, pv_validation)
    daylight = pv_validation["is_daylight_forecast"].eq(1)
    validation_rows[model_name] = {
        "Corrected RMSE (p.u.)": rmse(pv_validation["solar_power_pu"], corrected_forecast),
        "Corrected MAE (p.u.)": mae(pv_validation["solar_power_pu"], corrected_forecast),
        "Daylight RMSE (p.u.)": rmse(
            pv_validation.loc[daylight, "solar_power_pu"],
            corrected_forecast.loc[daylight],
        ),
        "Raw negative values": int((raw_forecast < 0).sum()),
        "Raw positive night values": int(
            (raw_forecast.loc[~daylight] > 0).sum()
        ),
    }

pv_validation_comparison = pd.DataFrame(validation_rows).T
selected_pv_model = pv_validation_comparison["Corrected RMSE (p.u.)"].idxmin()
display(pv_validation_comparison.round(3))
print("Selected using validation RMSE:", selected_pv_model)
"""


PV_CONSTRAINT_STUDENT = """def apply_pv_constraints(point_forecast, forecast_data):
    \"\"\"Enforce the 0–1 p.u. range and zero output at forecast night.\"\"\"
    # TODO 6.1: return a corrected copy with the same index.
    raise NotImplementedError("Apply the PV physical constraints.")


def rmse(actual, forecast):
    return np.sqrt(np.mean((np.asarray(actual) - np.asarray(forecast)) ** 2))


def mae(actual, forecast):
    return np.mean(np.abs(np.asarray(actual) - np.asarray(forecast)))


raw_validation_forecasts = {
    "Pooled interactions": pooled_validation_raw,
    "24 hourly models": hourly_validation_raw,
}

# TODO 6.2: for each candidate, calculate corrected full-day RMSE and MAE,
# daylight-only RMSE, and counts of raw negative and positive night values.
validation_rows = {}

if not validation_rows:
    raise NotImplementedError("Build the PV validation comparison table.")

pv_validation_comparison = pd.DataFrame(validation_rows).T
selected_pv_model = pv_validation_comparison["Corrected RMSE (p.u.)"].idxmin()
display(pv_validation_comparison.round(3))
print("Selected using validation RMSE:", selected_pv_model)
"""


PV_REFIT_CODE = """# Selection used only the validation period. Refit the selected formulation on
# all data available before calibration, then leave calibration and test untouched.
pv_point_fit_data = pd.concat([pv_train, pv_validation])

if selected_pv_model == "Pooled interactions":
    final_pv_model = LinearRegression()
    final_pv_model.fit(
        pv_point_fit_data[POOLED_PV_FEATURES],
        pv_point_fit_data["solar_power_pu"],
    )
    pv_calibration_raw = pd.Series(
        final_pv_model.predict(pv_calibration[POOLED_PV_FEATURES]),
        index=pv_calibration.index,
    )
    pv_test_raw = pd.Series(
        final_pv_model.predict(pv_test[POOLED_PV_FEATURES]),
        index=pv_test.index,
    )
else:
    final_pv_model = fit_hourly_linear_models(
        pv_point_fit_data, HOURLY_PV_FEATURES
    )
    pv_calibration_raw = predict_hourly_linear_models(
        final_pv_model, pv_calibration, HOURLY_PV_FEATURES
    )
    pv_test_raw = predict_hourly_linear_models(
        final_pv_model, pv_test, HOURLY_PV_FEATURES
    )

pv_calibration_point = apply_pv_constraints(pv_calibration_raw, pv_calibration)
pv_test_point = apply_pv_constraints(pv_test_raw, pv_test)

pv_test_daylight = pv_test["is_daylight_forecast"].eq(1)
pv_point_summary = pd.DataFrame({
    "Raw linear forecast": {
        "Full-day RMSE (p.u.)": rmse(pv_test["solar_power_pu"], pv_test_raw),
        "Full-day MAE (p.u.)": mae(pv_test["solar_power_pu"], pv_test_raw),
        "Daylight RMSE (p.u.)": rmse(
            pv_test.loc[pv_test_daylight, "solar_power_pu"],
            pv_test_raw.loc[pv_test_daylight],
        ),
        "Negative values": int((pv_test_raw < 0).sum()),
        "Positive night values": int(
            (pv_test_raw.loc[~pv_test_daylight] > 0).sum()
        ),
    },
    "Physically constrained": {
        "Full-day RMSE (p.u.)": rmse(pv_test["solar_power_pu"], pv_test_point),
        "Full-day MAE (p.u.)": mae(pv_test["solar_power_pu"], pv_test_point),
        "Daylight RMSE (p.u.)": rmse(
            pv_test.loc[pv_test_daylight, "solar_power_pu"],
            pv_test_point.loc[pv_test_daylight],
        ),
        "Negative values": int((pv_test_point < 0).sum()),
        "Positive night values": int(
            (pv_test_point.loc[~pv_test_daylight] > 0).sum()
        ),
    },
}).T

display(pv_point_summary.round(3))
"""


PV_INTERVAL_MARKDOWN = """### Task 7 — Why one interval for all 24 hours fails for PV

Use September–October residuals from the physically constrained point forecast to form 90% intervals for November–December.

First apply one pair of residual quantiles to every hour. This is a deliberately poor PV practice: the resulting interval has the same additive width at midnight and noon. Then compare it with residual quantiles estimated separately for each hour of day.

Finally, apply the PV constraints to **both** bounds. At night, setting only the lower bound to zero is not enough: the complete interval must collapse to $[0,0]$. Report coverage and width for both the full day and daylight hours. Full-day coverage can look deceptively good because the many zero night-time observations are easy to cover.
"""


PV_INTERVAL_SOLUTION = """def apply_pv_interval_constraints(lower, upper, forecast_data):
    corrected_lower = lower.clip(lower=0, upper=1.0)
    corrected_upper = upper.clip(lower=0, upper=1.0)
    night = forecast_data["is_daylight_forecast"].eq(0)
    corrected_lower.loc[night] = 0
    corrected_upper.loc[night] = 0
    return corrected_lower, corrected_upper


pv_calibration_residuals = (
    pv_calibration["solar_power_pu"] - pv_calibration_point
)

pv_constant_lower_raw, pv_constant_upper_raw = constant_residual_interval(
    pv_test_point, pv_calibration_residuals, alpha=ALPHA
)
pv_hourly_lower_raw, pv_hourly_upper_raw = hourly_residual_interval(
    pv_test_point, pv_calibration_residuals, alpha=ALPHA
)

pv_constant_lower, pv_constant_upper = apply_pv_interval_constraints(
    pv_constant_lower_raw, pv_constant_upper_raw, pv_test
)
pv_hourly_lower, pv_hourly_upper = apply_pv_interval_constraints(
    pv_hourly_lower_raw, pv_hourly_upper_raw, pv_test
)
"""


PV_INTERVAL_STUDENT = """def apply_pv_interval_constraints(lower, upper, forecast_data):
    \"\"\"Constrain both interval bounds and collapse night intervals to zero.\"\"\"
    # TODO 7.1: return corrected lower and upper Series.
    raise NotImplementedError("Apply the PV constraints to both interval bounds.")


pv_calibration_residuals = (
    pv_calibration["solar_power_pu"] - pv_calibration_point
)

# Reuse your Task 1 functions. Do not estimate residual quantiles on pv_test.
pv_constant_lower_raw, pv_constant_upper_raw = constant_residual_interval(
    pv_test_point, pv_calibration_residuals, alpha=ALPHA
)
pv_hourly_lower_raw, pv_hourly_upper_raw = hourly_residual_interval(
    pv_test_point, pv_calibration_residuals, alpha=ALPHA
)

pv_constant_lower, pv_constant_upper = apply_pv_interval_constraints(
    pv_constant_lower_raw, pv_constant_upper_raw, pv_test
)
pv_hourly_lower, pv_hourly_upper = apply_pv_interval_constraints(
    pv_hourly_lower_raw, pv_hourly_upper_raw, pv_test
)
"""


PV_QUANTILE_MARKDOWN = """### Task 8 — Transfer quantile regression to PV

Fit the 5th, 50th and 95th conditional quantiles using the selected pooled feature set. The pre-test data can all be used here: unlike a residual interval, direct quantile regression does not need a separate residual-calibration sample.

Linear quantile forecasts can cross and can violate PV limits. As in the load exercise, first order the three predicted quantiles at each timestamp. Then apply the same 0–1 p.u. and night-time constraints to every quantile. Compare the resulting 90% interval with the residual-based alternatives.
"""


PV_QUANTILE_SOLUTION = """pv_quantile_fit_data = pd.concat([
    pv_train, pv_validation, pv_calibration
])
pv_raw_quantiles = pd.DataFrame(index=pv_test.index)
pv_quantile_models = {}

for quantile in QUANTILES:
    model = make_pipeline(
        StandardScaler(),
        QuantileRegressor(quantile=quantile, alpha=0.01, solver="highs"),
    )
    model.fit(
        pv_quantile_fit_data[POOLED_PV_FEATURES],
        pv_quantile_fit_data["solar_power_pu"],
    )
    pv_quantile_models[quantile] = model
    pv_raw_quantiles[quantile] = model.predict(pv_test[POOLED_PV_FEATURES])

pv_crossing_mask = (
    (pv_raw_quantiles[0.05] > pv_raw_quantiles[0.50])
    | (pv_raw_quantiles[0.50] > pv_raw_quantiles[0.95])
)
print("PV timestamps with crossing raw quantiles:", pv_crossing_mask.sum())

pv_ordered_quantiles = pd.DataFrame(
    np.sort(pv_raw_quantiles.to_numpy(), axis=1),
    index=pv_test.index,
    columns=QUANTILES,
)
pv_quantiles = pd.DataFrame(index=pv_test.index)
for quantile in QUANTILES:
    pv_quantiles[quantile] = apply_pv_constraints(
        pv_ordered_quantiles[quantile], pv_test
    )

pv_quantile_scores = pd.Series({
    "QS q=0.05": quantile_score(
        pv_test["solar_power_pu"], pv_quantiles[0.05], 0.05
    ),
    "QS q=0.50": quantile_score(
        pv_test["solar_power_pu"], pv_quantiles[0.50], 0.50
    ),
    "QS q=0.95": quantile_score(
        pv_test["solar_power_pu"], pv_quantiles[0.95], 0.95
    ),
    "90% interval score": interval_score(
        pv_test["solar_power_pu"],
        pv_quantiles[0.05],
        pv_quantiles[0.95],
        alpha=ALPHA,
    ),
})
display(pv_quantile_scores.round(3).to_frame("Score"))
"""


PV_QUANTILE_STUDENT = """pv_quantile_fit_data = pd.concat([
    pv_train, pv_validation, pv_calibration
])
pv_raw_quantiles = pd.DataFrame(index=pv_test.index)
pv_quantile_models = {}

for quantile in QUANTILES:
    # TODO 8.1: fit a StandardScaler + QuantileRegressor pipeline using
    # POOLED_PV_FEATURES, then predict pv_test.
    pass

if set(pv_quantile_models) != set(QUANTILES):
    raise NotImplementedError("Fit all three PV quantile-regression models.")

# TODO 8.2: identify raw quantile crossings, order each row, and apply
# apply_pv_constraints to each ordered quantile.
pv_crossing_mask = ...
pv_quantiles = ...

if pv_crossing_mask is ... or pv_quantiles is ...:
    raise NotImplementedError("Order and physically constrain the PV quantiles.")

print("PV timestamps with crossing raw quantiles:", pv_crossing_mask.sum())
pv_quantile_scores = pd.Series({
    "QS q=0.05": quantile_score(
        pv_test["solar_power_pu"], pv_quantiles[0.05], 0.05
    ),
    "QS q=0.50": quantile_score(
        pv_test["solar_power_pu"], pv_quantiles[0.50], 0.50
    ),
    "QS q=0.95": quantile_score(
        pv_test["solar_power_pu"], pv_quantiles[0.95], 0.95
    ),
    "90% interval score": interval_score(
        pv_test["solar_power_pu"],
        pv_quantiles[0.05],
        pv_quantiles[0.95],
        alpha=ALPHA,
    ),
})
display(pv_quantile_scores.round(3).to_frame("Score"))
"""


PV_INTERVAL_SUMMARY = """pv_interval_variants = {
    "Constant residual — raw": (pv_constant_lower_raw, pv_constant_upper_raw),
    "Constant residual — physical": (pv_constant_lower, pv_constant_upper),
    "Hourly residual — raw": (pv_hourly_lower_raw, pv_hourly_upper_raw),
    "Hourly residual — physical": (pv_hourly_lower, pv_hourly_upper),
    "Linear quantile — physical": (pv_quantiles[0.05], pv_quantiles[0.95]),
}

pv_interval_rows = {}
for method_name, (lower, upper) in pv_interval_variants.items():
    pv_interval_rows[method_name] = {
        "Full-day coverage (%)": empirical_coverage(
            pv_test["solar_power_pu"], lower, upper
        ),
        "Full-day mean width (p.u.)": mean_interval_width(lower, upper),
        "Daylight coverage (%)": empirical_coverage(
            pv_test.loc[pv_test_daylight, "solar_power_pu"],
            lower.loc[pv_test_daylight],
            upper.loc[pv_test_daylight],
        ),
        "Daylight mean width (p.u.)": mean_interval_width(
            lower.loc[pv_test_daylight], upper.loc[pv_test_daylight]
        ),
        "Night mean width (p.u.)": mean_interval_width(
            lower.loc[~pv_test_daylight], upper.loc[~pv_test_daylight]
        ),
        "Negative lower bounds": int((lower < 0).sum()),
        "Positive night upper bounds": int(
            (upper.loc[~pv_test_daylight] > 0).sum()
        ),
    }

pv_interval_summary = pd.DataFrame(pv_interval_rows).T
display(pv_interval_summary.round(3))
"""


PV_INTERVAL_PLOTS = """pv_interval_week = pv_test.loc["2013-12-02":"2013-12-08 23:00"].index

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
interval_examples = [
    (
        "Bad practice: one unconstrained interval for every hour",
        pv_constant_lower_raw,
        pv_constant_upper_raw,
    ),
    (
        "Hour-dependent interval with PV physical constraints",
        pv_hourly_lower,
        pv_hourly_upper,
    ),
]

for ax, (title, lower, upper) in zip(axes, interval_examples):
    ax.fill_between(
        pv_interval_week,
        lower.loc[pv_interval_week],
        upper.loc[pv_interval_week],
        alpha=0.3,
        label="90% prediction interval",
    )
    ax.plot(
        pv_interval_week,
        pv_test.loc[pv_interval_week, "solar_power_pu"],
        color="black",
        linewidth=1.4,
        label="Actual PV",
    )
    ax.plot(
        pv_interval_week,
        pv_test_point.loc[pv_interval_week],
        color="tab:orange",
        label="Point forecast",
    )
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.set(ylabel="Normalised PV power (p.u.)", title=title)
    ax.legend(loc="upper right")

axes[1].set_xlabel("Valid time (UTC)")
plt.tight_layout()
plt.show()

width_by_hour = pd.DataFrame({
    "Constant — raw": pv_constant_upper_raw - pv_constant_lower_raw,
    "Hourly — raw": pv_hourly_upper_raw - pv_hourly_lower_raw,
    "Hourly — physical": pv_hourly_upper - pv_hourly_lower,
    "Linear quantile — physical": pv_quantiles[0.95] - pv_quantiles[0.05],
}).groupby(pv_test.index.hour).mean()

width_by_hour.plot(figsize=(10, 4), marker="o")
plt.xlabel("Hour of day (UTC)")
plt.ylabel("Mean interval width (p.u.)")
plt.title("PV prediction-interval width by hour")
plt.xticks(range(0, 24, 2))
plt.tight_layout()
plt.show()
"""


PV_SCENARIO_MARKDOWN = """### Task 9 — Transfer Gaussian trajectories to PV

The load-scenario functions can also be applied to 24-hour PV residual vectors. Non-negative projection alone is still insufficient: a Gaussian draw may remain positive at night. Apply the 0–1 p.u. range and forecast-daylight mask to every scenario after sampling.

The independent and multivariate constructions retain their earlier interpretation. Physical projection makes trajectories feasible, but it does not create realistic temporal dependence; the multivariate residual covariance is still needed for coherent daylight errors.

The original forecast trajectories run from 01:00 to 00:00 UTC. Because the sites are in Australia, that window contains the end of one daylight period and the beginning of the next. For this visualisation, reorganise both the calibration residuals and the selected test data into 15:00–14:00 UTC windows. This spans one continuous Australian solar day: night, sunrise, daytime, sunset and night. It combines valid times from two consecutive forecast origins, but every predictor remains the weather forecast associated with its own valid time.
"""


PV_SCENARIO_CODE = """def apply_pv_scenario_constraints(scenarios, forecast_day):
    corrected = np.clip(scenarios.copy(), 0, 1.0)
    night_columns = forecast_day["is_daylight_forecast"].eq(0).to_numpy()
    corrected[:, night_columns] = 0
    return corrected


SOLAR_DAY_START_HOUR_UTC = 15

# Shift the grouping boundary to 15:00 UTC and keep only complete 24-hour
# windows. This makes each residual vector follow one Australian solar day.
pv_solar_day_key = (
    pv_calibration_residuals.index
    - pd.Timedelta(hours=SOLAR_DAY_START_HOUR_UTC)
).floor("D")
pv_complete_solar_day = (
    pv_calibration_residuals.groupby(pv_solar_day_key)
    .transform("size")
    .eq(24)
)
pv_calibration_residual_matrix = (
    pv_calibration_residuals.loc[pv_complete_solar_day]
    .to_numpy()
    .reshape(-1, 24)
)

pv_scenario_start = pd.Timestamp("2013-12-04 15:00", tz="UTC")
pv_scenario_index = pd.date_range(pv_scenario_start, periods=24, freq="h")
pv_scenario_frame = pv_test.loc[pv_scenario_index]
pv_point_trajectory = pv_test_point.loc[pv_scenario_frame.index].to_numpy()

pv_independent_nonnegative = independent_gaussian_scenarios(
    pv_point_trajectory,
    pv_calibration_residual_matrix,
    n_scenarios=100,
    random_state=42,
)
pv_multivariate_nonnegative = multivariate_gaussian_scenarios(
    pv_point_trajectory,
    pv_calibration_residual_matrix,
    n_scenarios=100,
    random_state=42,
)

pv_independent_scenarios = apply_pv_scenario_constraints(
    pv_independent_nonnegative, pv_scenario_frame
)
pv_multivariate_scenarios = apply_pv_scenario_constraints(
    pv_multivariate_nonnegative, pv_scenario_frame
)

scenario_night = pv_scenario_frame["is_daylight_forecast"].eq(0).to_numpy()
pv_scenario_checks = pd.DataFrame({
    "Independent Gaussian": {
        "Positive night values before mask": int(
            (pv_independent_nonnegative[:, scenario_night] > 0).sum()
        ),
        "Positive night values after mask": int(
            (pv_independent_scenarios[:, scenario_night] > 0).sum()
        ),
    },
    "Multivariate Gaussian": {
        "Positive night values before mask": int(
            (pv_multivariate_nonnegative[:, scenario_night] > 0).sum()
        ),
        "Positive night values after mask": int(
            (pv_multivariate_scenarios[:, scenario_night] > 0).sum()
        ),
    },
}).T
display(pv_scenario_checks)

fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, sharey=True)
pv_plot_hours = range(15, 39)
pv_tick_positions = [15, 18, 21, 24, 27, 30, 33, 36, 38]
pv_tick_labels = [f"{hour % 24:02d}:00" for hour in pv_tick_positions]

for ax, (title, scenario_array) in zip(
    axes,
    [
        ("Independent Gaussian PV scenarios", pv_independent_scenarios),
        ("Multivariate Gaussian PV scenarios", pv_multivariate_scenarios),
    ],
):
    for scenario in scenario_array[:20]:
        ax.plot(pv_plot_hours, scenario, color="tab:blue", alpha=0.18)
    ax.plot(
        pv_plot_hours,
        pv_scenario_frame["solar_power_pu"],
        color="black",
        linewidth=2,
        label="Actual PV",
    )
    ax.plot(
        pv_plot_hours,
        pv_point_trajectory,
        color="tab:orange",
        linewidth=2,
        label="Point forecast",
    )
    ax.axvline(24, color="0.5", linestyle="--", linewidth=0.8)
    ax.set(ylabel="Normalised PV power (p.u.)", title=title)
    ax.legend(loc="upper right")

axes[1].set(
    xlabel="UTC hour (00:00–14:00 are on the following date)",
    xticks=pv_tick_positions,
    xticklabels=pv_tick_labels,
)
plt.tight_layout()
plt.show()
"""


STUDENT_QUESTIONS = """## 8. Interpretation questions

1. How do the constant and hour-dependent load intervals differ in coverage and width?
2. Which of the three load interval methods has the best interval score on the test period?
3. Why must residual quantiles come from calibration data rather than test residuals?
4. What does the coverage-by-lead plot reveal that overall coverage hides?
5. Why do independently sampled Gaussian scenarios look less coherent than multivariate Gaussian scenarios?
6. What limitation follows from estimating a 24-dimensional covariance matrix from only 31 calibration days?
7. Which linear PV formulation was selected, and what evidence supports the choice?
8. Why should physical constraints still be applied when the raw point-forecast violations are small?
9. Why does one constant residual interval give a misleading picture of PV uncertainty at night?
10. Why should full-day and daylight-only PV coverage both be reported?
11. Why is MAPE unsuitable for a complete PV time series?
12. Why must physical constraints be applied to every predicted quantile and every scenario?
13. Does physical projection remove the need to model temporal dependence in PV scenarios?

### Your answers

1. 
2. 
3. 
4. 
5. 
6. 
7. 
8. 
9. 
10. 
11. 
12. 
13. 
"""


SOLUTION_QUESTIONS = """## 8. Interpretation questions

1. **How do the constant and hour-dependent load intervals differ in coverage and width?**  
   The constant interval covers 81.25% of test observations with a mean width of 624.17 MW. The hourly interval is sharper at 515.88 MW, but its coverage falls to 74.26%, so the extra flexibility does not correct the July-to-August calibration shift in this example.

2. **Which of the three load interval methods has the best interval score on the test period?**  
   Linear quantile regression has the lowest interval score at 1011.83. Its 89.14% coverage is close to the nominal 90% and its mean width is 788.13 MW. The score balances sharpness with penalties for misses rather than rewarding coverage alone.

3. **Why must residual quantiles come from calibration data rather than test residuals?**  
   Test residuals contain the outcomes used for final evaluation. Using them to construct the intervals would leak test information and make coverage look too favourable.

4. **What does the coverage-by-lead plot reveal that overall coverage hides?**  
   Overall coverage can average together hours with undercoverage and hours with overcoverage. The lead plot identifies parts of the day where intervals are too narrow, too wide or systematically biased.

5. **Why do independently sampled Gaussian scenarios look less coherent than multivariate Gaussian scenarios?**  
   Independent sampling allows the error to jump without regard to neighbouring hours. Its mean adjacent-hour residual correlation is about 0.02, whereas the multivariate scenarios reproduce the calibration value of about 0.97 and therefore create more persistent deviations.

6. **What limitation follows from estimating a 24-dimensional covariance matrix from only 31 calibration days?**  
   The estimate is noisy and sensitive to unusual days. A longer calibration period, covariance regularisation or a lower-dimensional dependence model would be more stable.

7. **Which linear PV formulation was selected, and what evidence supports the choice?**  
   The 24 separate hourly models were selected because they have the lower physically constrained validation RMSE. Their daylight RMSE is also lower, and they produce far fewer positive night-time values before projection. The result is plausible because the relationship between forecast irradiance and normalised power changes substantially across the daily solar cycle.

8. **Why should physical constraints still be applied when the raw point-forecast violations are small?**  
   Even small violations are physically infeasible and may propagate into later decisions or probabilistic forecasts. The 0–1 p.u. projection and forecast-radiation daylight mask guarantee feasible outputs without presenting the otherwise accurate point forecast as a deliberately poor example.

9. **Why does one constant residual interval give a misleading picture of PV uncertainty at night?**  
   It adds the same residual offsets at every hour, producing negative lower bounds and positive upper bounds when true night-time PV is effectively fixed at zero. The physical interval should collapse to $[0,0]$ at night.

10. **Why should full-day and daylight-only PV coverage both be reported?**  
    Night contains many easy zero observations and can inflate full-day coverage. Daylight-only coverage reveals whether the interval represents uncertainty during the hours when PV power and forecast errors matter.

11. **Why is MAPE unsuitable for a complete PV time series?**  
    The denominator is zero at night and can be very small around sunrise or sunset. MAPE is therefore undefined or dominated by low-output hours.

12. **Why must physical constraints be applied to every predicted quantile and every scenario?**  
    Correcting only the point forecast does not stop interval bounds or sampled trajectories from becoming negative, exceeding 1 p.u. or remaining positive at night. Every reported probabilistic outcome must belong to the same physical support as PV power.

13. **Does physical projection remove the need to model temporal dependence in PV scenarios?**  
    No. Projection enforces feasibility but says nothing about how errors persist between neighbouring daylight hours. Independent scenarios can still jump unrealistically, whereas the multivariate model transfers the calibration residual covariance.
"""


REFERENCES = """## References

- ECMWF Parameter Database. https://codes.ecmwf.int/grib/param-db/
- Gneiting, T. and Katzfuss, M. (2014) ‘Probabilistic forecasting’, *Annual Review of Statistics and Its Application*, 1, pp. 125–151.
- Hong, T. and Fan, S. (2016) ‘Probabilistic electric load forecasting: A tutorial review’, *International Journal of Forecasting*, 32(3), pp. 914–938.
- Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001
- Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A. and Hyndman, R.J. (2016) ‘Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond’, *International Journal of Forecasting*, 32(3), pp. 896–913. https://doi.org/10.1016/j.ijforecast.2016.02.001
- Koenker, R. and Bassett, G. (1978) ‘Regression quantiles’, *Econometrica*, 46(1), pp. 33–50.
- Van der Meer, D.W., Widén, J. and Munkhammar, J. (2018) ‘Review on probabilistic forecasting of photovoltaic power production and electricity consumption’, *Renewable and Sustainable Energy Reviews*, 81, pp. 1484–1512.
"""


def source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines(source),
    }


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def update_notebook(path: Path, solution: bool) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if len(notebook["cells"]) < 27:
        raise ValueError(f"Unexpected notebook structure in {path}")

    notebook["cells"][0]["source"] = source_lines(INTRODUCTION)
    imports = "".join(notebook["cells"][2]["source"])
    imports = imports.replace(
        "from sklearn.linear_model import QuantileRegressor, Ridge",
        "from sklearn.linear_model import LinearRegression, QuantileRegressor, Ridge",
    )
    notebook["cells"][2]["source"] = source_lines(imports)

    # Preserve user edits outside the generated sections. Remove a previously
    # generated calibration pair before rebuilding, so this script is safe to
    # run repeatedly.
    pv_start = next(
        (
            position
            for position, cell in enumerate(notebook["cells"])
            if cell.get("id") == "pv-overview"
        ),
        23,
    )
    prefix_cells = [
        cell
        for cell in notebook["cells"][:pv_start]
        if cell.get("id")
        not in {"load-coverage-calibration-explanation", "load-coverage-calibration"}
    ]

    quantile_markdown_position = next(
        position
        for position, cell in enumerate(prefix_cells)
        if cell["cell_type"] == "markdown"
        and "Task 3" in "".join(cell["source"])
        and "Linear quantile regression" in "".join(cell["source"])
    )
    quantile_code_position = next(
        position
        for position, cell in enumerate(prefix_cells)
        if cell["cell_type"] == "code"
        and "QUANTILES =" in "".join(cell["source"])
        and "raw_quantile_predictions" in "".join(cell["source"])
    )
    prefix_cells[quantile_markdown_position]["source"] = source_lines(
        LOAD_QUANTILE_MARKDOWN
    )
    prefix_cells[quantile_code_position]["source"] = source_lines(
        LOAD_QUANTILE_SOLUTION if solution else LOAD_QUANTILE_STUDENT
    )
    prefix_cells[quantile_code_position]["execution_count"] = None
    prefix_cells[quantile_code_position]["outputs"] = []

    prefix_cells[quantile_code_position + 1:quantile_code_position + 1] = [
        markdown_cell(
            LOAD_COVERAGE_MARKDOWN,
            "load-coverage-calibration-explanation",
        ),
        code_cell(LOAD_COVERAGE_CODE, "load-coverage-calibration"),
    ]

    pv_cells = [
        markdown_cell(PV_OVERVIEW, "pv-overview"),
        code_cell(PV_LOAD_CODE, "pv-load"),
        code_cell(PV_EXPLORATION_CODE, "pv-explore"),
        markdown_cell(PV_POINT_MARKDOWN, "pv-point-task"),
        code_cell(PV_FEATURE_CODE, "pv-features"),
        code_cell(PV_MODEL_SOLUTION if solution else PV_MODEL_STUDENT, "pv-models"),
        markdown_cell(PV_CONSTRAINT_MARKDOWN, "pv-constraints-task"),
        code_cell(
            PV_CONSTRAINT_SOLUTION if solution else PV_CONSTRAINT_STUDENT,
            "pv-constraints",
        ),
        code_cell(PV_REFIT_CODE, "pv-refit"),
        markdown_cell(PV_INTERVAL_MARKDOWN, "pv-interval-task"),
        code_cell(
            PV_INTERVAL_SOLUTION if solution else PV_INTERVAL_STUDENT,
            "pv-intervals",
        ),
        markdown_cell(PV_QUANTILE_MARKDOWN, "pv-quantile-task"),
        code_cell(
            PV_QUANTILE_SOLUTION if solution else PV_QUANTILE_STUDENT,
            "pv-quantiles",
        ),
        code_cell(PV_INTERVAL_SUMMARY, "pv-interval-summary"),
        code_cell(PV_INTERVAL_PLOTS, "pv-interval-plots"),
        markdown_cell(PV_SCENARIO_MARKDOWN, "pv-scenario-task"),
        code_cell(PV_SCENARIO_CODE, "pv-scenarios"),
        markdown_cell(
            SOLUTION_QUESTIONS if solution else STUDENT_QUESTIONS,
            "interpretation",
        ),
        markdown_cell(REFERENCES, "references"),
    ]
    notebook["cells"] = prefix_cells + pv_cells

    if not solution:
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    update_notebook(STUDENT_PATH, solution=False)
    update_notebook(SOLUTION_PATH, solution=True)
    print("Updated Tutorial 4.2 student and solution notebooks.")
