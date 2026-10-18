# Tutorial 4.2: forecast uncertainty

The main example uses Robin_education_Margarito (UCL), 2017. A separate GEFCom2014
Zone 3 PV example follows. The records are from different locations and years and
are not combined into measured net demand.

## Building data

Source: `data/week4_day1/robin/robin_education_margarito_2017.csv`.
SHA256: `8c1868e10e5158643928481bfc9df81596b2a0069de95347e3e3d93023214f3b`.

There are 8760 unique hourly timestamps. Electricity readings are positive and
complete. Seven measured temperatures and 24 hot-water readings are missing;
neither variable is used for forecasting. No source value is changed or filled.

Electricity is **kWh per hourly interval**, not instantaneous kW. Sum the hourly
readings to obtain daily kWh without multiplying by duration again. Timestamps
have no timezone offsets; retain the labels provided, without conversion or a
daylight-saving adjustment. Readings are assumed available at their labelled time.

Margarito is a pseudonym. The source comes from cleaned BDG2 tables; completeness
does not imply that the upstream readings were never cleaned.
See the [Robin data notes](../week4_day1/robin/README.md) and provenance JSON.
Attribution and the CC BY-SA 4.0 licence are retained.

Miller et al. (2020), ‘The Building Data Genome Project 2, energy meter data from
the ASHRAE Great Energy Predictor III competition’, *Scientific Data*, 7, 368.
[Paper](https://doi.org/10.1038/s41597-020-00712-x),
[dataset](https://github.com/buds-lab/building-data-genome-project-2).

## Training, calibration and testing

| Role | Inclusive target dates | Rows |
|---|---|---:|
| Lag history | 1–7 January 2017 | 168 |
| Training | 8 January–30 June 2017 | 4176 |
| Calibration | 1 July–30 September 2017 | 2208 |
| Test | 1–28 October 2017 | 672 |

The notebook trains `StandardScaler` and `Ridge(alpha=10)` on January–June only.
Inputs are lag 24, lag 168, hour sine/cosine, weekday sine/cosine and a weekend
indicator. Lags are constructed before splitting on the full hourly grid.
The model is trained once and used for calibration and test forecasts.

All 24 targets for one day share the previous-day 23:00 origin. Earlier observed
test values can supply predictors for later days, but the model is not retrained.
Negative point forecasts are projected to zero; raw predictions are retained.

## Prediction intervals and scenarios

Calibration error is actual minus the nonnegative point forecast. Main prediction
intervals use the 10th and 90th error quantiles, estimated either from all 2208
calibration errors or from 92 errors per target hour. There are no fallback groups
or catch-up flags. Endpoints are projected to zero without an arbitrary upper cap.

Coverage is inclusive and returned as a percentage. Width and Winkler are in kWh.
Scoring retains all 672 original test targets, including 480 weekdays and 192 weekends.
Metric functions reject invalid lengths, indices, non-finite inputs and reversed bounds.
The supplied sensitivity compares 50%, 80% and 95% nominal coverage on identical targets.

Daily calibration errors form a 92-by-24 matrix. Each row contains the errors of one
complete forecast issued from a common origin. For 8 October, add each row once to
the point profile. The resulting 92 scenarios have equal weights. A second ensemble
independently permutes each hourly column, preserving hourly samples but changing
cross-hour dependence. Daily energy is the row sum in kWh.

The scenario example assumes that summer error patterns remain relevant in October.
One target day does not validate joint calibration or demonstrate operational savings.

## PV example

Source: `data/tutorial_4_2/gefcom2014_solar_zone3.csv`.
SHA256: `ab2326d07dc82d2f2da78050c08534eb67aaabbacfc5ec4a2baa272595f921ac`.

The notebook constructs the inputs and trains `LinearRegression` directly; it does
not require saved Day 1 models or predictions. Features are lag-1 PV, forecast
irradiance (W/m²), forecast air temperature (°C) and forecast cloud fraction.
Weather must have been issued by the one-hour-ahead origin; immediate availability
is assumed. Timestamps are UTC; the exact site/local timezone is undisclosed.
Forecast irradiance has already been converted to W/m².

Training: April 2012–March 2013, 8759 rows before the lag and 8758 retained.
Calibration: April–June 2013, 2184 rows. Test: July–September 2013, 2208 rows.
See the notebook for exact inclusive endpoints.

The raw target is preserved alongside a teaching copy bounded to $[0,1]$.
The one raw exceedance, 1.00355 at 4 November 2013 02:00 UTC, is outside all selected
periods. No calibration or test actual is changed. Point forecasts are clipped to
$[0,1]$ and set to zero when forecast irradiance is at most 1 W/m². This is an
imperfect forecast-based proxy, not an exact indicator of astronomical night.
Errors are calculated after point-forecast corrections.

Compare bounded pooled intervals with a variant additionally set to zero under
this proxy. Retain all actuals and report full-test and subset scores.
[Original PV data preparation](../tutorial_4_2/README.md).

Hong et al. (2016), ‘Probabilistic energy forecasting: Global Energy Forecasting
Competition 2014 and beyond’, *International Journal of Forecasting*, 32(3), 896–913.
[Paper](https://doi.org/10.1016/j.ijforecast.2016.02.001).

## Self-study extensions

1. Fit linear 10th/90th quantile models on the same building training data, with
   prespecified L1 regularisation 0.1. Retain raw quantiles, resolve crossing and
   project bounds to zero before comparing scores on the same test targets.
2. Fit one pooled normal error distribution and a 24-dimensional multivariate
   normal model from calibration data only. Draw 500 scenarios per model, seed 42.
   Compare trajectories, dependence and daily energy; Gaussian errors are an assumption.
3. Estimate PV prediction intervals by UTC target hour using 91 calibration errors
   per group. Compare with the pooled intervals using identical support clipping.

The extensions are outside the 200-minute lecture. No parameter is selected using
test results. Ordering or support corrections do not guarantee nominal coverage.

## Reproduction and outputs

Generated tables, plots, manifests and validation reports are excluded from Git. The current notebooks read the source datasets directly; saved predictions are not required. Worked solutions are released separately.

Start Jupyter from the repository root. Each notebook is self-contained.

```bash
python -m jupyter nbconvert --to notebook --execute --inplace tutorial_4_2_forecast_uncertainty_solution.ipynb --ExecutePreprocessor.timeout=900
python teaching/validate_week4_day2.py
```

The solution writes the U01–U19 lecture figures, a separate filled U11, and extension
figures E01–E03 as 220 dpi PNG and vector PDF. The manifest, instructor notes and
validation summary are under `teaching/`. Tables and predictions in this directory
are computed outputs, not source data; `run_metadata.json` records splits, inputs,
versions, source checksums and correction counts.

Obsolete catch-up bundles and their rebuilding helper were moved to
`teaching/week4_day2_archive/removed_catch_up_2026-10-13/`. They are not dependencies
of the current notebooks. The earlier PV-first revision remains in the same archive
under `pv_first_2026-10-13/`.
