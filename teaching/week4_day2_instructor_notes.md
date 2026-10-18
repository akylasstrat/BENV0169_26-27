# Tutorial 4.2: instructor notes

Use `tutorial_4_2_forecast_uncertainty` and its solution. The running example is
Robin_education_Margarito, 2017. We train Ridge on January–June, estimate calibration
errors on July–September and evaluate on October. The PV section trains its linear
model directly; it does not load a saved model or depend on Tutorial 4.1b execution.

The revision retains the lecturer's wording and plotting changes. It uses
**calibration errors**, explains pooled versus hour-conditioned intervals, separates
training and plotting into short cells, and removes catch-up flags and small-group
fallback machinery. Detailed verification is kept in the instructor validator.

## Timing and lecture map

The core has 200 teaching minutes, with an additional 15–20-minute break after
minute 100 / slide 20. Total elapsed time is 215–220 minutes.

| Teaching minutes | Slides | Sections | Activity |
|---|---|---|---|
| 0–50 | 1–12 | U0–U2 | Model, calibration errors, representations; A (9+3 min) |
| 50–100 | 13–20 | U3–U4 | Hour conditioning and scores; B (9+3), C (7+3) |
| Extra break | After 20 | — | 15–20 minutes outside teaching time |
| 100–150 | 21–27 | U4–U7 | Comparison, D (8+4), pinball and daily scenarios |
| 150–162 | 28 | U7 | E (9+3 min) |
| 162–187 | 29–32 | U8 | Supplied PV and F (6+4), 25 minutes total |
| 187–200 | 33–35 | Recap | Findings, limitations and references |

Per-slide minutes for slides 1–35:
`2,2,4,3,1,5,5,3,5,4,4,12,5,6,12,1,6,4,6,10,8,12,6,1,8,8,7,12,1,6,8,10,9,2,2`.

The core retains five code exercise cells in A/B/C/E. D/F require interpretation.
Extensions add five code exercises for independent study, not in-class completion.
Students should complete them after the lecture. Extension 1 implements the quantile
regression discussed in hidden slide 40; Extension 2 implements normal intervals,
covariance and scenarios related to hidden slides 36–38. Monitoring on slide 39
remains discussion of future recalibration and subsequent evaluation.

For slides 17–18, use the exported U09/U10 figures before the metric task if following
the slide order. Live score plotting follows completion of Task C in the notebook.

## Run and maintain

```bash
python -m jupyter nbconvert --to notebook --execute --inplace tutorial_4_2_forecast_uncertainty_solution.ipynb --ExecutePreprocessor.timeout=900
python teaching/validate_week4_day2.py
```

Neither notebook imports an external configuration module or needs saved answers.
`build_week4_day2_notebooks.py` is an instructor authoring utility, not a runtime
dependency. Preserve manual notebook edits before rebuilding. The current sources
include the lecturer's reviewed edits.

The former catch-up bundle and rebuilding helper are recoverable in
`week4_day2_archive/removed_catch_up_2026-10-13/`. They are absent from the active
teaching path. Keep the solution, filled U11 and these notes until the intended
solution release. No assessment, Day 1 source or PowerPoint is changed.

## Numerical and interpretation checkpoints

### U0–U1: model and calibration errors

Training/calibration/test counts are 4176/2208/672. Electricity is positive and
complete; measured temperature and hot-water missing values do not remove targets.
Use the timestamp labels provided. Readings are kWh per hourly interval.

Scaler and Ridge learn from training only. Projection to zero is post-processing,
not another fit. No building point forecast or main interval endpoint is negative
in this run. Calibration error is actual minus forecast. Its pooled 10th/90th
percentiles are **−44.891 / 46.043 kWh**, nearly but not exactly symmetric.

### A–B: pooled and hour-conditioned intervals

Pooled intervals use all 2208 calibration errors and have raw width **90.934 kWh**.
U05 shows the two error offsets with arrows at one target time, rather than an inset.
Hour-conditioned intervals use 92 errors per hour, without fallback flags.
Hour 0 offsets are −23.462/36.654 kWh; hour 10 offsets are −48.630/61.115 kWh.
The error spread differs by hour, but conditioning need not improve every test score.

### C: scores

For the constructed $[400,500]$ kWh interval and actuals 380/450/530:
coverage flags are False/True/False, widths are 100 each and Winkler scores are
300/100/400 kWh. Mean coverage is 33.333%; mean Winkler is 266.667 kWh.
Endpoint equality incurs no miss penalty.

| Method | Subset | Coverage (%) | Width (kWh) | Winkler (kWh) |
|---|---|---:|---:|---:|
| Pooled | All 672 | 75.595 | 90.934 | 132.784 |
| Target-hour | All 672 | 74.256 | 89.153 | 130.720 |
| Pooled | Weekdays 480 | 72.500 | 90.934 | 141.182 |
| Target-hour | Weekdays 480 | 69.375 | 89.153 | 141.540 |
| Pooled | Weekends 192 | 83.333 | 90.934 | 111.788 |
| Target-hour | Weekends 192 | 86.458 | 89.153 | 103.669 |

Both methods under-cover overall at the nominal 80% level. Target-hour intervals
have lower overall width/Winkler but lower coverage; weekday Winkler is slightly
worse. Each hourly test statistic uses only 28 cases, which may be dependent.

### D and U6: coverage level and pinball loss

Nominal 50/80/95% gives empirical coverage 47.024/74.256/94.643% and mean widths
47.959/89.153/147.899 kWh. Increasing nominal coverage widens the intervals.
Do not select a level using this test set or compare raw Winkler across different
miscoverage parameters as if the penalty were unchanged.

For $e=y-q$, high quantiles penalise positive errors more strongly.
The unscaled pinball loss at the median equals half MAE; FPP's quantile score
uses an additional factor two.

### E: daily scenarios

Pivot calibration errors by date/hour to form a 92-by-24 matrix. Every row belongs
to one forecast issued from a common origin. For the 8 October target, origin
7 October 23:00, add each daily error vector to the point profile once.

Shuffling columns preserves hourly values and quantiles but changes dependence.
Both daily means are **11455.819 kWh**. Standard deviations are **653.927 kWh** for
whole vectors and **203.371 kWh** after column shuffling. Mean cross-hour correlations
are 0.486167 and 0.008805. The shuffled sample correlation need not be exactly zero.
Daily energy is a sum of hourly kWh, without another duration factor.

### F: PV

Training/calibration/test counts after the lag are 8758/2184/2208.
The supplied linear regression uses lag-1 PV and three weather forecasts issued by
the origin. Clip predictions to $[0,1]$ and use forecast irradiance >1 W/m² as a
generation-availability proxy. Do not change actuals based on this proxy.

Pooled bounded intervals have all-hours coverage **75.453%**, width **0.162 p.u.**
and Winkler **0.361 p.u.** Adding the zero rule gives **70.969%**, **0.088 p.u.**
and **0.287 p.u.** The proxy is imperfect: 99 positive actuals occur among 1164
false-proxy test hours. The zero rule misses these observations.

## Self-study results

- **Linear quantile regression:** coverage 73.810%, width 127.010 kWh and Winkler
  178.114 kWh. It performs worse than the two error-based intervals here.
  Do not assume that a more direct or more complex approach must perform better.
  The L1 regularisation parameter 0.1 is specified in advance, not test-tuned.
- **Normal pooled interval:** coverage 79.315%, width 97.705 kWh and Winkler
  131.870 kWh. This result does not establish normality or transferability.
  Scenario panels show independent pooled and multivariate draws on the same day.
  Their comparison changes both hourly distributions and dependence; Task E is
  the controlled dependence comparison.
- **PV hour conditioning:** all-hours coverage 79.937%, width 0.096 p.u. and Winkler
  0.200 p.u. Forecast-solar-available coverage is only **62.452%**, despite an
  all-hours score close to 80%. Night-time zeros can conceal daytime undercoverage.

## Figures and limitations

The existing U01–U19 slide filenames are retained. U11 has separate blank and filled
versions. E01 quantile regression, E02 Gaussian scenarios and E03 PV hourly intervals
are self-study figures with no assigned lecture slide. All have 220 dpi PNG and
vector PDF counterparts. Use the manifest for exact slide numbers, data periods,
units and source checksums; inspect the contact sheet alongside full-size figures.

Common mistakes: learning from test data; reversing error sign; treating corrected
forecasts as newly fitted models; imposing a $[0,1]$ bound on building energy;
multiplying hourly kWh by duration again; equating marginal coverage with joint
calibration; or treating the irradiance proxy as exact astronomical night.

Limitations remain cleaned source readings, summer-to-October error transfer,
small dependent hourly groups, one scenario target day, uncertain Gaussian
assumptions and an imperfect PV proxy. No coverage guarantee or control savings
are asserted.
