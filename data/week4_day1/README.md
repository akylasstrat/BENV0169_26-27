# Week 4, Day 1: teaching data preparation

The revised Lecture 4.1 uses the [Robin / UCL building data](robin/README.md) and the PV example below. The Alfonso preparation in this file documents the retained earlier version, not the current load route.

The two standalone notebooks reuse the original repository extracts by relative path. They do not overwrite them, download additional data or use assessment datasets. The building and PV records represent independent sites and years. Do not combine their values into net building demand.

## Building-meter example

Source: `data/tutorial_4_1/building_data_genome_sample.csv`, BDG2 **Rat_education_Alfonso**, a K–12 education building. Original source description: [Tutorial 4.1 data](../tutorial_4_1/README.md).

The BDG2 building metadata and cleaned meter files provide only electricity for this building. No gas, hot-water, chilled-water, steam, water, irrigation or solar meter series is provided. This indicates missing coverage in the dataset, not zero consumption. The supplied meter summary and initial electricity plot make this limitation explicit; the selected building is unchanged.

| Column | Meaning |
|---|---|
| `timestamp` | Supplied hourly building clock, without an explicit UTC offset |
| `electricity_kwh` | Energy in an hourly meter interval, kWh |
| `air_temperature_c` | Observed site temperature, °C; descriptive plots only |

The source contains 17,544 unique hourly timestamps, 1 January 2016 00:00–31 December 2017 23:00. There are no absent timestamps, 79 stored electricity NaNs, 11 weather NaNs and no electricity zeros. Preserve the supplied clock grid. Although the source metadata identifies US/Eastern, this tutorial does not silently localise it or change daylight-saving hours. Readings are assumed available at their labelled timestamp for teaching.

The dates were fixed after checking data quality, before comparing models:

- Lag history: 1–7 January 2016, 168 hours.
- Training targets: 8 January 00:00–30 September 23:00, 6,408 grid rows.
- Test targets: 1 October 00:00–28 October 23:00, 672 complete, positive observed targets from 28 daily origins.
- Forecast plot: 8–14 October 2016. Its 168 cases are a display subset, not the full numerical evaluation.
- Resolution plot: seven complete training days, 8–14 February 2016.

The training target at 13 March 2016 02:00 is missing. Its cause is unknown. Preparation copies the **observed** value at 6 March 02:00 (264.090 kWh) into a separate training-history copy. If the previous week's value is missing, use the training mean for the same weekday and hour, then the overall training mean if needed. No fallback was needed in this execution. All mean estimates use 8 January–30 September only. Observed values and the record of missing readings are retained.

This reconstructed history supplies two lag inputs. The unobserved training target itself is excluded, leaving **6,407 fitting rows**. Training targets are not converted into fabricated measurements. Test targets/history are unchanged. All 672 test rows have originally observed lag-24/168 sources, so no test exclusion or input fallback is needed. There are 480 weekday and 192 weekend test cases. The descriptive temperature scatter uses 6,400 original observed pairs, excluding eight incomplete pairs; profiles exclude the reconstructed reading.

The nine predictors are `load_lag_24`, `load_lag_168`, `hour`, `day_of_week`, `is_weekend`, `hour_sin`, `hour_cos`, `day_sin`, `day_cos`. Hour period is 24, weekday period 7, Monday=0. Lags are constructed before any rows are dropped. At the preceding day's 23:00 origin, load lags 24/168 and all target-calendar fields are known. Lag 1 is unavailable for 23 of the next 24 targets. Ridge uses cyclic calendar inputs and training-only scaling; the forest uses integer calendar inputs without scaling. Neither model uses temperature in the core exercise.

MAPE requires strictly positive actuals. The selected test is positive. If the dates are changed to include nonpositive actuals, the supplied upstream guard labels MAPE undefined while retaining those targets for MAE/RMSE. Metrics reject misaligned or non-finite pairs rather than silently removing them.

Building source SHA256: `77eba7fa9c946aac2ab09c660b67494b08975780ec058c00f53473ead47c24fc`.

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. (2020), ‘The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition’, *Scientific Data*, 7, 368. https://doi.org/10.1038/s41597-020-00712-x. The source is distributed under CC BY-SA 4.0; the extract's provenance and licence remain applicable.

## PV example

Source: `data/tutorial_4_2/gefcom2014_solar_zone3.csv`, Australian **GEFCom2014 solar Zone 3**. [Original provenance and weather mapping](../tutorial_4_2/README.md).

| Column | Meaning |
|---|---|
| `timestamp_utc` | Target/weather valid timestamp, explicitly UTC-labelled |
| `solar_power_pu` | Source realised capacity-normalised PV power, p.u. |
| `forecast_surface_solar_radiation_w_m2` | Forecast hourly mean surface irradiance, W/m² |
| `forecast_air_temperature_c` | Forecast air temperature, °C |
| `forecast_total_cloud_cover_fraction` | Forecast cloud cover, fraction |
| `forecast_issue_time_utc` | UTC weather-forecast issue timestamp |
| `forecast_lead_hours` | Source weather lead, retained for metadata, not a model feature |

The original extract contains 19,704 unique, complete hourly timestamps, 1 April 2012 01:00–1 July 2014 00:00 UTC, and no missing required values. Exact plant coordinates are undisclosed. The UTC interpretation follows the extract provenance; we do not infer an exact Australian local timezone from it.

The irradiance column has **already** been derived by differencing accumulated radiation **within each daily forecast trajectory**, treating lead 1 as the first accumulation since the origin, then dividing by 3,600 seconds. Never difference across origin resets or difference this flux again. Source accumulated fields remain in the original extract for audit.

The Day 1 notebooks derive a bounded teaching target in a separate in-memory copy, retaining `solar_power_pu_raw`. Exactly one source target changes: **4 November 2013 02:00 UTC, 1.003550 to 1.000000 p.u.** It is outside all selected fitting/calibration/test periods. The rationale is an explicit [0,1] teaching convention; nominal-capacity exceedances in the source are not treated as a universal physical impossibility. Raw and bounded targets agree at all evaluated timestamps. Persistence, lag construction, fitting and scoring use the same bounded target. Original files are unchanged.

| Target period (UTC) | Before lag removal | Retained |
|---|---:|---:|
| Train: 1 April 2012 01:00–31 March 2013 23:00 | 8,759 | 8,758 |
| Calibration: 1 April–30 June 2013, inclusive | 2,184 | 2,184 |
| Test: 1 July–30 September 2013, inclusive | 2,208 | 2,208 |

Only the initial training row lacks a pre-extract lag-1 observation. No calibration/test target is discarded. The training profile uses 10–12 May 2012 UTC, and the fixed test plot 15–21 July 2013 UTC.

Each one-hour-ahead forecast has origin = target minus one hour. All weather issue timestamps satisfy issue ≤ origin. The source has no release-latency field, so the tutorial explicitly assumes zero release latency and instantaneous availability of the previous labelled PV observation. A real application must verify these assumptions.

The four model inputs, in order, are `pv_lag_1`, `forecast_surface_solar_radiation_w_m2`, `forecast_air_temperature_c`, `forecast_total_cloud_cover_fraction`. Students fit linear regression, the decision tree and the random forest once on training, then predict calibration/test without refitting. Constructors and hyperparameters are supplied. All models use the same four inputs and prediction corrections. All four methods, including the worked persistence example, appear in the comparison. No calendar features are used in this PV example; the encoding explanation appears only in Part a. The Day 2 handover model remains linear regression, chosen in advance rather than by final test ranking.

Clip forecasts to [0,1], then zero them when forecast irradiance ≤1 W/m². Apply this rule to both persistence and regression and retain raw/corrected predictions. The `solar_available` flag is an operational forecast proxy, not astronomical certainty. In test, 1,164 targets have a false proxy; 99 of these have a small positive actual target. Those actuals remain unchanged and scored. Do not force truth to agree with the correction. PV MAPE is not calculated.

PV source SHA256: `ab2326d07dc82d2f2da78050c08534eb67aaabbacfc5ec4a2baa272595f921ac`.

Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A. and Hyndman, R.J. (2016), ‘Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond’, *International Journal of Forecasting*, 32(3), 896–913. https://doi.org/10.1016/j.ijforecast.2016.02.001. Weather parameter definitions: ECMWF Parameter Database, https://codes.ecmwf.int/grib/param-db/.

## Derived artefacts and release

`day2_handover/` contains aligned calibration/test predictions, metadata and a standalone rebuilding cell. It does not contain assessment data or intervals. See its [release and rebuilding instructions](day2_handover/README.md).

Optional intermediate feature/forecast/score CSVs in `teaching/week4_day1_checkpoints/` are generated from the executed solutions. They let the instructor move the class forward after an unfinished exercise. They are worked teaching outputs, not assessed benchmark answers. Release selectively at the corresponding classroom checkpoint. Figures include source, period, units and relevant counts; the manifest ties them to notebook checkpoints and actual slide numbers.
