# Tutorial 6.2 instructor notes

The pair now follows I0–I8 with six core tasks A–F and optional E1–E4.
Four 50-minute blocks include the 64 minutes of task/debrief time; the break
after slide 16 is an additional 15–20 minutes. Optional real-data work is outside
the 200-minute core. Figures are supplied after the calculations.

This revision retains the prescribed synthetic core. The real-house demonstration
is independently runnable and the extension is implemented, so it can support a
later decision to use real data in the core. No dataset switch or downstream
controller-model change has been made.

## Main corrections and evidence

- Split observations before constructing transitions. Fit on 1007 transitions;
  validate on 335 common targets. Exclude 26 January 23:30 → 27 January 00:00.
- Fit-period OLS and bounded coefficients agree: b=0.023558345,
  c=0.125662197 K/kW. Neither bound is active. Do not invent a failed OLS example.
- Effective R=5.334084 K/kW, C=3.978921 kWh/K, tau=21.223902 h.
  The no-intercept Euler model omits gains and fabric dynamics. Noise enters both
  regressors and differences; plausible coefficients do not establish unbiased
  or uniquely physical parameters.
- Common-target RMSE: one-step persistence 0.516286 K, one-step fitted 0.220008 K,
  seven-day free rollout 1.481329 K. The information sets differ. The initial
  observation is unscored; bias always means measured minus predicted.
- Six complete midnight origins are used for every horizon. Pooled window RMSE
  is 0.997915, 1.354214 and 1.478824 K over 6, 12 and 24 h. Lead error is not
  monotonic; inspect its daily structure. Scores reuse recorded future inputs
  and are conditional model diagnostics, not operational forecast accuracy.
- An origin is the prediction start, a lead is elapsed time to a target, and a
  horizon is the evaluated window length. Origins are once daily, 00:00 on
  27 January–1 February. Each 48-step rollout is initialised once from the
  measured state; coefficients remain fixed and no reset occurs at 6 or 12 h.
  The illustrative table shows the first origin ending at 06:00, 12:00 and
  28 January 00:00, with 12, 24 and 48 targets.
- F62-09 compares two information protocols on identical origin/target rows.
  Blue uses updated measured states for repeated one-step predictions, not a
  24 h forecast issued at midnight. Orange uses only the midnight indoor state.
  Their first lead agrees. The left panel reports RMSE across six origins at
  each lead; the right retains rollout RMSE over every target in each first-H-hour
  window. Neither error must grow monotonically. New benchmark columns retain
  the original rollout fields; `one_step_benchmark_lead_scores.csv` adds the
  matched lead summary without replacing `lead_scores.csv` or `window_scores.csv`.
- Residual records retain predictor and target times. The residual at k+1 is
  paired with useful heat and outdoor temperature at k.
  For 10:00→10:30, compare the temperature error at 10:30 with heating input
  during that interval. Input associations suggest model diagnostics, not proof
  that the plotted input caused the error. Positive residual means underprediction.

Full precision is retained in `week6_day2_outputs/`. The supplied generator was
read in the original local reference checkout without executing it: its air
measurement/process standard deviations are 0.06/0.012°C, with process noise one
third as large for fabric. The source records the initial measurement before the
first step. Those are simulator settings, not estimated uncertainty parameters.

## Real-house demonstration and extension

The frozen files match the local BENV0092 reference copies byte-for-byte.
They contain 720 ten-minute observations each: 11–15 November 2016 and
10–14 June 2017. Source and teaching references:
[UCL dataset](https://discovery.ucl.ac.uk/id/eprint/10087216/) and
[BENV0092 Exercise 3](https://github.com/akylasstrat/BENV0092/blob/main/exercise_3_RC_model.ipynb).
No download is needed at runtime. The original data README and files are unchanged.

The audit finds one missing indoor value in each period, and 19/23 missing
outdoor values. The longest outdoor runs are one/three samples respectively.
Main fitting/scoring uses originally observed finite transitions: 698 November
and 694 June. The fixed November model uses electricity-plus-gas power in kW and
solar in kW/m²; it does not identify delivered space heat. The solar coefficient
has units K per (kW/m²) in this discrete regression.

Missing forcing intervals end rollouts; 20 November and 21 June segments are
explicitly recorded and reset from measurements. Missing temperature targets do
not themselves reset the trajectory. Longest supported segments are 22.833/24.167 h.
On common original targets, one-step RMSE is 0.048647/0.033929 K and segmented
rollout RMSE is 0.437188/0.676807 K for November/June. These are not five-day
fixed-origin forecast scores.
The one-step horizon is ten minutes. November is the fitting period, so its
scores are in-sample diagnostics; June is evaluated with unchanged November
coefficients before refitting. Close one-step traces can coexist with rollout
drift. Since segment lengths differ, F62-10 does not directly compare common
6/12/24 h horizons; use F62-09 for that experiment.

E1 tests a declared maximum-three-sample internal-gap interpolation assumption.
Endpoint/long gaps cannot be filled, and interpolated temperature targets never
become scored truth. The optional continuous rollout fills outdoor forcing only
and scores the same original targets. Its November/June RMSE is 0.996526/4.946156 K.
Filling the forcing gaps also joins segments and removes resets: the comparison
changes both input assumptions and initialisation, not interpolation accuracy alone.
The extra plot makes this distinction visible.

E2 reproduces the original-observation November fit and applies it unchanged to
June. E3 follows that evaluation with descriptive full/first/final-November and
post-hoc June fits. Both first/final 48 h are clock-time windows, with 278 valid
transitions each. Their effective R values are 10.871610/3.036314 K/kW and C values
24.436532/72.453542 kWh/K. The June refit is labelled post-hoc/in-sample and never
replaces the frozen model or headline scores. E4 discusses proxy limitations,
gains, identifiability and additional measurements.
The supplied display reports rounded R,C,tau alongside counts and active bounds
from the computed E3 table. In the existing fits, all bounds are inactive. Input
variation, correlations, unmodelled gains and noise differ across periods;
effective parameter variation does not itself demonstrate changing building
fabric. Inspect the different relative changes in R,C and tau rather than
assuming every parameter changes substantially. Four fits support no significance
or confidence claims. The grey bar is labelled “June refit (post-hoc)”.

To explore only real data, run setup, plotting helpers, the metric-function cell
and I7. A fresh-kernel test confirmed that this path needs neither synthetic
tasks nor optional completions. Promoting it to the core later requires updating
task timing, ten-minute/three-coefficient units, the missing-input protocol and
slides together; the common Week 7 model should remain a separately documented role.

## Figures, model provenance and validation

The figure manifest contains ten core IDs F62-01–10 and optional F62-11, each
with 220-dpi PNG and text-preserving SVG. Two extra notebook figures show raw
house signals and the input-gap sensitivity. All canonical figures were reviewed
against the current 36-slide UCL deck, inspected read-only. Preserve the full-width
placements; residual diagnostics use a single row of panels to keep labels
readable in the shallow content slot. No slide was edited.
The synthetic overview retains the user's stacked layout. The raw-house overview
retains all six panels and original signals, with short date labels only on the
bottom row. `week6_day2_supplemental_placements.csv` records its separate placement
on slide 27 and source hashes, without adding a canonical figure ID.

The deck has core slides 1–31, references on 32 and optional work on 33–36.
Task positions are A7/B13/C15/D19/E21/F25, with unchanged 12/12/8/12/8/12-minute
allocations (64 minutes). The canonical figure map is F62-01→4, 02→5, 03→6
(referenced on 8), 04→14, 05→18, 06→20, 07→22, 08→24, 09→26, 10→28 and
11→35 (optional). Transfer the former slide-28 model-role explanation to slide
29's notes; the corresponding notebook provenance and common model remain intact.

`week6_day2_schedule.csv` records the per-slide core allocations:

| Slides | Minutes in order | Total |
|---|---|---:|
| 1–9 | 3,3,6,5,6,5,12,5,5 | 50 |
| 10–16 | 6,5,5,12,6,8,8 | 50 |
| 17–24 | 5,5,12,6,8,4,5,5 | 50 |
| 25–31 | 12,6,8,8,6,5,5 | 50 |

These are planning allocations, not measured delivery times. The break is
15–20 additional minutes after slide 16; references and optional work are extra.

The common `verified_controller_model.json` retains its original schema and
values. Its generator fits all 28 synthetic days; it is not the student holdout
fit. Separate output `model_provenance.json` records both roles and the real-house
protocol. This revision resolves the 6.2 issues recorded in the earlier day-1
notes; those protected day-1 materials were not rewritten.

Run from the repository root in the existing pinned environment:

```text
.venv/Scripts/python.exe -B scripts/build_week6_day2.py
.venv/Scripts/python.exe -B scripts/validate_week6_day2.py
```

Reconcile live edits before rebuilding; rebuilding clears solution outputs.
Validation executes the solution, a private completed student copy and the
standalone house demonstration in fresh kernels. Independent checks cover known
coefficient recovery, bound activation, perturbed measurement information sets,
metric signs, negative timing cases, complete horizon aggregation, original masks,
segment resets, restricted interpolation and post-hoc separation. The student
outputs remain empty. The protected-hash baseline for this revision was captured
before editing, including the user's intervening day-1 edits. The historical
baseline and pre-edit outputs are retained in the private revision snapshot;
this is distinct from validating against the earlier, stale day-1 hashes.
All 138 current protected hashes must match, including day 1, Week 4,
assessment files, shared data/API and the common JSON. Interpreter/packages,
commands and private evidence location are recorded in `validation_results.json`.
Use `--baseline-dir <private-pre-edit-output-directory>` to compare every original
exported table column with its pre-edit value, and `--refresh-figures F62-01
F62-08 F62-09 F62-10 F62-11` to retain unaffected figure files byte-for-byte while
refreshing the affected exports and raw-house overview. Private checks also test
the benchmark formula, matched first lead and measurement perturbations.

Placement checklist for the PowerPoint editor (the deck is not edited here):

| Slide | Export | Placement |
|---|---|---|
| 24 | `w6d2_F08_residual_diagnostics.png` | Existing three-panel area; interval labels and residual sign |
| 26 | `w6d2_F09_horizon_validation.png` | Existing two-panel area; repeated one-step versus midnight rollout |
| 27 | `extra_house_raw_signals.png` | Six original signal panels; bottom-row dates only |
| 28 | `w6d2_F10_measured_house_transfer.png` | Full width below two bullets; preserve aspect ratio and reset markers |
| 35 | `w6d2_F11_fit_period_sensitivity.png` | Optional sensitivity figure; June refit remains post-hoc |

Implementation checks pass. Scientific adequacy remains conditional on the
operating range, horizon, heat-proxy validity and evaluation information set.
