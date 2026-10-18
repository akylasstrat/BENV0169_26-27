# Week 4, Day 1: instructor notes

## Current teaching route

Use `tutorial_4_1a_building_load.ipynb` and `tutorial_4_1b_pv_point_forecasting.ipynb`, with their `_solution` versions for worked outputs. Part a uses Robin/UCL. The earlier smart-meter forecasting notebooks are retained as legacy material. Week 1, assessments, source data and Tutorial 4.2 are unchanged.

The revised lecture contains **45 slides**. Follow its revised slide content: several older speaker notes and placeholder filenames still refer to Alfonso, four load models or hourly boxplots. The PowerPoint is unchanged. Generic 4.1a labels in it refer to the Robin notebook for this session.

The load notebook has eight exercises: daily totals, weekday/weekend means, duration sorting, two demand indicators, training-only scaling, and fitting Ridge, decision tree and random forest. Feature construction and the weekly seasonal benchmark remain supplied. Each model has its own cell, a supplied constructor, fixed hyperparameters and short fit/predict TODOs. PV has six exercises: lag construction, linear fitting, linear corrections, tree fitting, forest fitting and the metric table. Tree/forest corrections and repetitive evaluation remain supplied. The calendar-encoding explanation and figure appear only in the Robin notebook. PV uses lagged output and forecast weather, with no calendar features.

The subsequent notebook revision adds tree and forest comparisons beyond the deck's two-model load/PV examples. The forecast figures now show all four methods. No PowerPoint changes were made.

## Run and release

Run from the repository root in the existing environment. Notebooks are independent and self-contained. Student files have no outputs or worked answers. Solutions regenerate 220 dpi PNGs and SVGs. Inline plots use 100 dpi. Optional result tables are exported to `teaching/week4_day1_outputs/`; neither notebook loads them.

```bash
python -m jupyter nbconvert --to notebook --execute --inplace tutorial_4_1a_building_load_solution.ipynb --ExecutePreprocessor.timeout=600
python -m jupyter nbconvert --to notebook --execute --inplace tutorial_4_1b_pv_point_forecasting_solution.ipynb --ExecutePreprocessor.timeout=600
python teaching/validate_week4_day1.py
```

The validator creates `week4_day1_lecture_figure_manifest.csv`, combining current Robin and PV figures and excluding Alfonso. Use it for the revised deck. Older images are retained but no longer selected by this map.

The authoring utility `build_week4_day1_notebooks.py` now contains the reviewed current notebook sources, not the older Alfonso template. It is not a runtime dependency. Review any subsequent manual notebook edits before rebuilding the pairs.

The teaching cells follow a single sequence: prepare inputs, train each model, predict, correct and evaluate. Catch-up flags and saved-answer branches have been removed. Detailed source, alignment, availability and metric tests are in `validate_week4_day1.py`, rather than repeated in the notebooks. The few notebook guards reject incomplete hourly indices, unavailable weather forecasts or invalid metric inputs. Keep solutions and worked outputs with instructor material until their intended release.

## Live teaching

Allow 70–90 minutes of load work and 35–45 minutes of PV work within the four-hour session, interleaved with slides and breaks.

| Slides | Notebook activity | Code/discussion time |
|---|---|---:|
| 6–13 | L0–L1: meter interpretation, quality, daily totals, privacy | 15 minutes |
| 16–20 | L2–L3: profiles, weather, duration, indicators, building uses | 25 minutes |
| 23–24 | L4: fixed group and coincident peaks | 15 minutes |
| 25–28 | L5–L8: supplied forecasts, diagnostic days, interpretation | 25 minutes |
| 31–33 | P0–P2: PV drivers, origin, fit and correction | 25 minutes |
| 34–35 | P3–P4: ramps, evaluation subsets and calibration/test roles | 15 minutes |

Ask the prediction question before each key plot. Afterwards ask for a result, evidence and explanation. Work through one model at a time, asking students to complete the fit/predict lines before moving on. Prioritise indicators, coincident peaks, predictor availability and the PV ramp/subset discussion. If time is short, demonstrate the remaining fit in the solution and ask students to complete their own version afterwards. The optional long trace can be skipped. Transfer, intervals and scenarios are not Day 1 exercises.

## Current figure map

Robin figures: `figures/week4_day1/robin/`. PV figures: `figures/week4_day1/`. Each has PNG/SVG counterparts. The current manifest records exact paths, source checksums, periods, units and counts.

| Figure | Slide | Section | Content |
|---|---:|---|---|
| F01, F18 | 6 | L0 | Six real rows and separate electricity/heat traces |
| F02 | 11 | L1 | Aligned hourly trace/daily totals, hourly peak and its day |
| F03 | 12 | L0 | Heat gap, available electricity and meter-specific audit |
| F04 | 16 | L2 | Weekday/weekend means and daily variation |
| F05 | 17 | L2 | Measured temperature against electricity and heat |
| F06 | 18 | L3 | Duration curve, maximum, median and 90th percentile |
| F07 | 19 | L3 | Calculated demand indicators |
| F19 | 20 | L3 | Usage profiles: 10 residences, 22 education, 16 offices |
| F20 | 23 | L4 | Original three-building readings and simultaneous sum |
| F21 | 24 | L4 | Separate peaks versus simultaneous peak |
| F08 | 25 | L5 | Origin, targets, predictors and chronological split |
| F09, F10 | 26 | L6 | Weekly naïve, Ridge, tree and forest: one week and full-test scores |
| F11 | 27 | L7 | Two daily forecasts with peak markers and diagnostics |
| F12 | 31 | P0 | Three training days of PV and forecast irradiance |
| F13 | 32 | P1 | Actual origin/issue/valid-time examples |
| F14, F15 | 34 | P3 | Weekly overview, ramp close-up and subset scores |
| F16 | 35 | P4 | Train once; separate training, calibration and test periods |
| F17, optional | 41 | Optional | 2017 same-building trace, causes unknown |
| F22 | 39 | L5 | Integer versus cyclical hour encoding, with actual feature rows in the notebook |

F06/F07/F08/F11 have new filenames because their meanings changed. Do not use the retained origin/calendar/split/boxplot images instead. Usage profiles now follow the single-building indicators.

## Verified Robin checkpoints

Margarito is a UCL college laboratory (8,468 m²). Its 2017 electricity is complete, with no identical-value run longer than two hours. Full-year missing counts: heat 24, temperature seven. Training missing counts: heat one, temperature seven. The natural heat gap is **20 August 2017 23:00**. Neither descriptive column is filled or used as a predictor. Electricity needs no imputation.

Retain the recorded Robin clock. Immediate availability of labelled readings is a teaching assumption. Training 8 January–30 September: **6,384** targets after **168** lag-history hours. Test 1–28 October: **672** complete positive observed targets. Every lag-24/168 donor is known at its prior-day 23:00 origin. The scaler and model fit training only and stay fixed.

Training indicators: load factor **0.624**, overnight energy share **22.4%**, weekend/weekday ratio **0.915**. Peak hourly average demand is approximately **863 kW**, at **21 June 2017 15:00**. These do not establish energy efficiency or waste. The temperature plots retain **6,377** electricity pairs and **6,376** heat pairs.

The fixed aggregation group is Margarito laboratory, Armand residence and Sammie office, chosen from metadata and training quality. All original readings for 6–12 February are present. Separate maxima occur at 6 February 12:00, 8 February 19:00 and 9 February 15:00. Their sum is **845.8 kW**, compared with the simultaneous maximum **797.5 kW** at 6 February 12:00. Weekly energy remains **108,295.4 kWh**. This is a selected group, not a documented shared feeder. No normalised profiles or zero-filled gaps are summed.

| Model | Full-test MAE, kWh | Full-test RMSE, kWh | n |
|---|---:|---:|---:|
| Weekly seasonal naïve | 38.474 | 47.512 | 672 |
| Ridge | 30.001 | 37.487 | 672 |
| Decision tree | 30.893 | 40.069 | 672 |
| Random forest | 29.128 | 37.516 | 672 |

Ridge RMSE is **21.1%** lower than the weekly benchmark and marginally lower than the forest. The forest has the lowest MAE. Both tree-based models use exactly the same integer/load predictors, without scaling. Ridge uses training-scaled cyclical features. Tree depth is fixed at 6, minimum leaf size 4; forest settings are 150 trees, minimum leaf size 4, max_features=0.8, seed 42. No setting is selected using final test scores. The encoding diagram illustrates the midnight boundary, not an empirical search over encodings.

The two diagnostic days remain the minimum/maximum daily Ridge MAE within the display week, selected for diagnosis only: **12 and 13 October**. On the 12th, Ridge has the lowest daily MAE (**11.900 kWh**) but the forest has the smaller peak error (**10.228 kW**, versus Ridge **10.750 kW**). On the 13th, the tree has the lowest daily MAE (**39.304 kWh**) and peak error (**28.256 kW**), despite worse full-test RMSE than Ridge or forest. Ridge, tree and forest place that day's peak at 14:00, two hours after the observed maximum. Full-period error, daily error, peak magnitude and timing answer different questions.

Usage profiles use observed training readings, excluding source NaNs and exact constant runs of at least 24 hours without changing the source. This conservative flag does not prove upstream imputation. Scale each building by its retained training maximum and weight building means equally. Compare shapes, not energy efficiency or absolute consumption.

## Verified PV checkpoints

Zone 3 is independent of UCL. Never combine them as measured net building demand. Source: **19,704** complete UTC rows. The bounded teaching copy clips one nominal exceedance, **1.003550 p.u. at 4 November 2013 02:00 UTC**, outside selected splits, and retains raw values. Irradiance is already converted to hourly forecast W/m². Do not difference it again or call it measured irradiance.

Training 1 April 2012 01:00–31 March 2013 23:00: **8,759** pre-lag, **8,758** fitted targets. Calibration April–June 2013: **2,184**. Test July–September: **2,208**. All calibration/test targets remain. Origin = target − one hour. Weather issue ≤ origin, assuming immediate publication because latency is unrecorded. Inputs are latest PV plus forecast irradiance, temperature and clouds valid for the target.

Tutorial 4.2 uses linear regression, a choice specified before test evaluation. Train once and reuse the model for calibration and test forecasts. Apply [0,1] clipping and the forecast-irradiance zero rule to all four models. Keep positive actual PV when the proxy is false. This proxy is not an exact indicator of astronomical night.

| Subset | Model | MAE, p.u. | RMSE, p.u. | n |
|---|---|---:|---:|---:|
| All test hours | Persistence | 0.059 | 0.110 | 2,208 |
| All test hours | Linear regression | 0.050 | 0.093 | 2,208 |
| Forecast irradiance >1 W/m² | Persistence | 0.125 | 0.160 | 1,044 |
| Forecast irradiance >1 W/m² | Linear regression | 0.105 | 0.135 | 1,044 |
| All test hours | Decision tree | 0.048 | 0.096 | 2,208 |
| Forecast irradiance >1 W/m² | Decision tree | 0.102 | 0.140 | 1,044 |
| All test hours | Random forest | 0.041 | 0.085 | 2,208 |
| Forecast irradiance >1 W/m² | Random forest | 0.087 | 0.124 | 1,044 |

The forest has the lowest MAE/RMSE in both PV evaluation subsets here. The tree and forest use identical four-feature inputs, the same training targets and the same corrections as persistence/regression. They fit training only and generate aligned calibration/test forecasts. Tutorial 4.2 trains its linear model directly from source data; it does not load a saved Day 1 model or prediction bundle.

The subset excludes many easy low-output hours and answers a different question. It is common to all models and uses no actual PV selection. The ramp close-up centres on **21 July 2013 06:00 UTC**, the largest observed one-hour change in the fixed display week. It is diagnostic, not a tuning criterion. Persistence is delayed and all learnt methods miss part of the drop. No unverified cause is asserted. UTC daytime is not local daytime.

Raw regression has **80** negative test values and none above one. The false-proxy subset contains **1,164** targets, including **99** with positive actual PV, retained in scoring. Day 2 uses calibration errors: actual minus the corrected linear forecast. The model settings, features, splits, correction rules and numerical scores are unchanged by the readability revision. No intervals or coverage are calculated here. The older handover files are retained but are not dependencies of these notebooks.

## Validation and common mistakes

The validator checks notebook syntax, cleared student outputs, paired supplied cells, intentional TODOs and clean-kernel execution. It completes exercises in memory without writing student answers. It independently checks indicators, group sums/peaks, daily peak metrics and PV subset scores, alongside target preservation, training-only scaling and lag/issue availability. Full-precision comparison confirms that the pre-revision forecast scores are unchanged. Source data, the older handover bundle and other tutorial notebooks are checked against recorded hashes. Figure layout is inspected separately.

Watch for averaging instead of summing daily energy, confusing hourly average and instantaneous power, summing normalised profiles as actual demand, interpreting coincidence as energy savings, treating missing heat as missing electricity, using future measured weather, fitting on test/calibration, and judging full-day PV scores without considering zero-output hours. Retain source-clock, availability and solar-proxy limitations.
