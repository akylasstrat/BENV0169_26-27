# Tutorial 7.1 instructor notes

This revision implements the bracketed comments saved in the solution notebook
on 18 October 2026. The paired notebooks and scoped builder are synchronised;
the source data, common model, other weeks, assessment and PowerPoint are unchanged.

The supplied draft now has 46 slides: core 1–42, references 43 and optional
44–46. Task slides are A 11, B 21, C 24, D 28, E 36 and F 40. The six tasks
retain 15/8/10/10/12/7 minutes, including debriefs. The schedule CSV retains
200 teaching minutes in four 50-minute blocks; the additional 15–20-minute
break follows slide 21. The learning-objectives and penalty-sensitivity slides
are included by redistributing explanatory time within the existing blocks.
Older numbers still present in the draft's speaker notes are not used as slide IDs.

## Changes requested in the notebook

- The graphical CVXPY example is split into clearly commented steps: declare
  variables, construct constraints, define the objective, solve and read results.
- Direct array calculations now precede the supplied result helpers. They show
  interval kWh, total kWh, the electricity bill, mean endpoint temperature,
  electrical peak kW and maximum half-hour interval energy.
- Detailed constraint-distance diagnostics, independent recurrence verification
  and the infeasibility fixture are private validation material. There is no
  reachability discussion or separate replay-verification section in the core.
- Rebuilding a flat-price problem and updating a price Parameter are both shown
  in small visible examples. The two approaches agree in optimal bill within
  tolerance; schedule equality is not required.
- The ToU, tariff and peak comparisons use the same nine common summary metrics.
  Own-tariff bills are shown alongside the distinct cross-evaluation table.
- The notation matches the slides: objective J_bill + gamma phi(s), with
  phi(s)=dt sum(s). Only the total-slack penalty is implemented in the core.
- The penalty sweep uses gamma=0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1, 1.5,
  2.5, 5, 10 and 25. The new F71-14 figure supports the sensitivity slide.
- The user's wording changes, simpler x/y axis labels and reference deletion
  are retained; editorial comments and empty scratch cells are removed.

## Scientific and teaching conventions

The building case remains the first 48 half-hour intervals on 3 February 2025.
There are 48 powers and 49 boundary temperatures, starting at 20°C. The common
instructor JSON was fitted on all 28 identification days, separately from the
Week 6 student holdout exercise. It is not refitted here.

Future inputs are known for this synthetic deterministic planning case. The
bill covers heat-pump electricity only. COP maps electrical power to useful
heat; the discrete coefficient c already contains the time step. Bill and
degree-hour calculations multiply by dt; the recurrence does not multiply
by dt again. Comfort applies at T[k+1]. Initial state and final comfort are
handled separately, with no additional terminal equality.

Fixed inputs do not rule out solver failure, coding mistakes or incorrect
indexing. Simple status, finite-value, dynamics and bound checks remain visible.
The longer independent checks are private and do not introduce model uncertainty
as another core teaching topic.

Mean indoor temperature is the arithmetic mean of the 48 predicted interval-end
states, excluding T0. Peak power is kW; maximum interval energy is peak power
times 0.5 h, in kWh. Shortfall means degrees below the lower band, overshoot
means degrees above the upper band. Their durations use a 1e-6°C counting
tolerance; degree-hour magnitudes retain all raw positive differences.
Endpoint compliance does not certify continuous comfort.

The own-tariff summary reports each schedule with its optimisation tariff.
It is a descriptive comparison that changes both prices and schedule.
The separate 2×2 table holds evaluation prices fixed within a column and
isolates scheduling value. For the current day the ToU saving from changing
the flat schedule is about £0.297. No strict saving or unique schedule is
assumed in general.

The expanded penalty sweep shows real thresholds and plateaus. In this case
the bill rises from about £4.330 at gamma=0.25 to the hard-comfort bill of
about £4.633 by gamma=1, within numerical tolerance. A smooth response is
not claimed. A finite weight does not universally guarantee hard comfort.
The penalty is an assigned modelling preference, not a metered charge.

Peak minimisation retains hard comfort and is measured in kW. The returned
minimum peak is about 0.966 kW; its bill and energy describe that returned
optimum, without a secondary tie-break objective.

## Tables, figures and optional material

Full-precision records are in [week7_day1_outputs](week7_day1_outputs/).

| Record | Use |
|---|---|
| hard_summary.csv | Single ToU plan, common nine-metric summary |
| tariff_summary.csv | Flat and ToU plans, each evaluated at its own prices |
| tariff_cross_costs.csv | Both fixed plans evaluated under both tariffs |
| F71-11_native_table.csv | Labelled 2×2 bill table for PowerPoint |
| peak_summary.csv | Cost/peak schedules with the same nine metrics |
| peak_comparison_metrics.csv | Full objective metadata and peak comparison |
| penalty_sensitivity.csv | All twelve weights, bill, slack, penalty and objective |

All twelve active figures have 12.5 × 3.5-inch canvases, 220-dpi PNGs and SVGs
with editable text. The figure manifest records slide/use, task, period, counts,
units, assumptions and source hashes.

| Figure | Slide/use | Content |
|---|---|---|
| F71-01 | 15 | Supplied planning inputs |
| F71-02 | 12 | Graphical LP debrief |
| F71-03 | 17 | Interval/boundary indexing |
| F71-04 | 25 | Hard ToU schedule |
| F71-05 | Notebook only | Hard/lower-soft traces |
| F71-06 | 37 | Bill versus shortfall |
| F71-08 | 29 | Flat/ToU schedule traces |
| F71-09 | 44, optional | Signed initial-observation error |
| F71-11 | 30 | Cross-evaluation bill table |
| F71-12 | 40 | Peak comparison and complete common metrics |
| F71-13 | 27 | Tariff inputs |
| F71-14 | 38 | Bill, objective and shortfall versus gamma |

F71-07 and F71-10 are retired from the active notebook material. Recoverable
copies and the original annotated notebook are in the private pre-edit snapshot.
The initial-observation-error extension remains supplied and optional. Its
schedule is fixed during replay from true 20°C. Core tables report predicted
mean temperature; the optional error records separately score realised comfort.
Quadratic tracking and terminal treatment remain optional discussion. No MPC
feedback loop is introduced.

## Reproduction and verification

Use the existing environment from the repository root:

~~~text
.venv/Scripts/python.exe -B scripts/build_week7_day1.py
.venv/Scripts/python.exe -B scripts/validate_week7_day1.py --snapshot <private-pre-edit-directory>
~~~

Building clears the solution outputs. Reconcile future manual edits before
rebuilding. Validation executes the solution and a privately completed student
copy in separate clean kernels using CLARABEL. It checks six informative student
stops, empty student outputs, deterministic source rebuilding, consistent summary
tables, both parameter-change methods, and independent graphical/hard/soft/peak
LP oracles. Independent recurrence, units, tariff optimality, all twelve penalty
cases, copied snapshots, solver failures and inaccurate status remain checked.

Current package versions, tolerances and results are in
[validation_results.json](week7_day1_outputs/validation_results.json).
The scope audit checks 491 protected repository files and the supplied draft
presentation against their pre-edit hashes. No commit, push or deck edit is made.
