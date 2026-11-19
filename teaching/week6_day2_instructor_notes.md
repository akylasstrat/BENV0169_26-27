# Tutorial 6.2 instructor notes

The tutorial retains the synthetic identification and validation sequence,
Tasks A–E, followed by the main real-house experiment, Tasks R1–R3. The former
Task F horizon aggregation and optional E1–E4 sequence have been removed.
Four 50-minute blocks include 52 minutes for A–E and 40 minutes for R1–R3;
the break after slide 16 adds 15–20 minutes. Slide numbers in the notebook and
manifests are revised placement suggestions. The PowerPoint deck requires a
separate update.

## Teaching aim

Students should explain how imperfect measurements and a simplified model
affect estimated parameters and prediction errors. The real-house exercise
focuses on the power-to-heat proxy, model simplification and the sensitivity of
the same 1R1C model to its training data. Connect the coefficient comparison
to the observed input ranges and omitted processes. Treat recovered values
as effective parameters for this model and dataset.

Fill missing readings with linear interpolation, separately within each
fitting and test period. This supplied preprocessing step supports the main
exercise. Keep discussion focused on the heat balance, coefficient differences
and transfer results. The original CSV files remain unchanged.

## Main experiment and expected findings

- R1 builds three predictors explicitly: outdoor-minus-indoor temperature,
  total power in kW and solar irradiance in kW/m². November contributes 719
  transitions. June is split at 13 June 2017 before interpolation and forming pairs:
  431 fitting transitions from 10–12 June and 287 test targets from 13–14 June.
  The boundary crossing is excluded.
- R2 fits the same three-coefficient model with the same bounds to November
  and June fitting data. November uses five days and June uses three. This
  is a descriptive training-dataset comparison, including differences in
  duration, input variation and occupancy conditions.
- R3 compares both frozen coefficient arrays and persistence on exactly the
  same June test targets. Each prediction uses the measured current indoor
  temperature. Both RC predictions use recorded interval forcing inputs.

The continuous energy balance adds effective solar heat gain
`g_eff * S`. With irradiance in kW/m², `g_eff` is in m²; it combines window
area, transmission, orientation and shading relative to the measured signal.
The Euler coefficients satisfy `g_eff = d/c`. Treat the heating coefficient
and recovered R,C under the electricity-plus-gas heat-proxy assumption.

Fresh execution gives:

| Quantity | November fit | June fit |
|---|---:|---:|
| b | 0.000800466 | 0.000950074 |
| c (K/kW) | 0.00532765 | 0.00198252 |
| d (K·m²/kW) | 0.0686476 | 0.0185166 |
| R_eff (K/kW) | 6.65569 | 2.08670 |
| C_eff (kWh/K) | 31.2833 | 84.0681 |
| Time constant (h) | 208.212 | 175.425 |
| g_eff (m²) | 12.8851 | 9.33996 |

All fitted coefficients lie inside their bounds. A change in an effective
parameter alone provides insufficient evidence of changed fabric or windows.
Occupancy, different heating use, correlated predictors, sensor errors and
omitted thermal states can change the least-squares approximation.

| June test method | N | MAE (K) | RMSE (K) | Bias (K) |
|---|---:|---:|---:|---:|
| November transfer | 287 | 0.0250143 | 0.0329398 | −0.0150921 |
| June refit | 287 | 0.0207287 | 0.0277248 | 0.00135561 |
| Persistence | 287 | 0.0229111 | 0.0309687 | 0.00338223 |

The June refit reduces RMSE and absolute bias on these held-out observations.
November transfer has slightly higher RMSE than persistence. Small one-step
errors partly reflect initialisation from the latest measured temperature.
Use the synthetic rollout comparison to discuss accumulated simulation error.
Further model changes require a new evaluation period.

## Synthetic example and validation

The synthetic fit remains b=0.023558345 and c=0.125662197 K/kW, giving
R=5.334084 K/kW, C=3.978921 kWh/K and a 21.223902 h time constant.
One-step RMSE remains 0.220008 K and seven-day rollout RMSE remains 1.481329 K.
The initial state is excluded from scoring. All seven retained pre-revision
synthetic output tables match their saved values to numerical tolerance.

Run from the repository root in the course environment:

```text
.venv/Scripts/python.exe -B scripts/build_week6_day2.py
.venv/Scripts/python.exe -B scripts/validate_week6_day2.py
```

Reconcile live notebook edits before rebuilding. The builder clears outputs;
validation writes executed outputs to the solution only. The student notebook
retains its TODOs, empty outputs and unset execution counts. Validation runs
the full solution, a privately completed student copy and the house section
independently in fresh kernels. Checks cover transition alignment, the hand
example, solar-unit conversion, known-coefficient recovery, model bounds,
shared scoring targets and unchanged coefficients after perturbing held-out
June temperatures. Synthetic timing, measurement-reset and metric-sign checks
remain in place. Evidence is recorded in
`teaching/week6_day2_outputs/validation_results.json`.

The revised figure manifest includes F62-01–08 and F62-10–12. F62-12 shows
raw house measurements, F62-11 compares seasonal coefficients and F62-10
compares June test predictions and residuals. Earlier ignored figure/table
exports can remain as historical files; the current notebook and manifest
identify the outputs used by this revision.
