# Tutorial 6.1 instructor notes

The revised pair follows M0–M7 and six interleaved tasks, A–F. Four 50-minute
teaching blocks include coding, inspection and debriefs; the 15–20-minute break
after slide 18 is additional. The tasks total 62 minutes within the 200 minutes.
Optional examples follow the recap and are outside the teaching total.

Use `week6_day1_task_manifest.csv` for slide numbers, timings and stable task-cell
IDs, and `week6_day1_figure_manifest.csv` for the ten canonical figures, exact
filenames, placements, data hashes, units, counts and captions. F61-04 is deliberately
reused on slides 13 and 19. Optional figures have separate names and are not extra
canonical manifest entries.

## Running and maintaining the material

Open the notebooks from the `BENV0169_26-27` repository root using its existing
module environment. All core calculations are visible in the notebook; plots are
supplied. The student version has six intentional TODOs and no saved outputs.
The solution is executed, including the separate optional examples. A privately
completed student copy was tested from a fresh kernel without importing answers
at runtime. Its path is recorded in `week6_day1_outputs/validation_results.json`.

The pasted checkout was missing `course_utils`. Its package initializer and
`reference_building.py` were copied byte-for-byte from the supplied local checkout.
No plant API or algorithm was changed. No external simulator, download or online
service is needed. Existing data and the common controller JSON are unchanged.

Authoring and validation commands from the repository root:

```text
.venv/Scripts/python.exe -B scripts/build_week6_day1.py
.venv/Scripts/python.exe -B scripts/validate_week6_day1.py
```

The builder writes only the 6.1 pair and task manifest. Reconcile live manual edits
before using it; it clears solution outputs, which validation then regenerates.
The older multi-week builder was not run. Core output CSVs are records of the
current calculations, never prerequisites for notebook execution.

## Data and numerical evidence

The inputs are synthetic. The core uses the first 96 half-hour operation intervals,
3 February 2025 00:00 to 5 February 00:00 at the final boundary. Compact simulations
store 97 boundary states. The parameter experiments use 192 intervals and 193
states over 96 hours. Temperatures are °C; differences K; power kW; time h;
resistance K/kW; capacitance kWh/K. Input CSVs distinguish interval starts/ends;
state CSVs use `state_time`. Heat-pump records include modulation, electrical power,
delivered heat, COP, duration and disturbances.

- Task A: useful heat 6 kW, electricity 1 kWh and useful thermal energy 3 kWh.
- Task B: next temperature 20.152777778°C for the specified analytic fixture.
- R sweep equilibria: 22.5, 32 and 45°C; time constants 14, 21.6 and 32 h.
- C sweep: common equilibrium 32°C; time constants 10.8, 21.6 and 43.2 h.
- Euler maximum absolute errors at each method's actual times: 0.060193 K
  (0.5 h), 0.517985 K (4 h), 6.164257 K (24 h). At 24 h, a = −0.111111:
  oscillatory and asymptotically stable. At 48 h, a = −1.222222: unstable.
- Noiseless reference versus compact air-temperature RMSE is 4.061402 K,
  excluding initialisation; maximum absolute discrepancy is 8.257001 K.
  Both use initial air 20°C, identical outdoor inputs and realised useful heat.
  Fabric starts at 20°C. The reference alone represents gains and separate fabric
  dynamics, so discrepancy includes omitted inputs and structure. This is not
  evidence of adequacy for a real building or of an optimised operating schedule.
- The reference experiment consumes 67.5 kWh of electricity and delivers
  209.635766 kWh of heat. Energy comes from interval power × duration, not a sum
  of temperatures. The optional noise run uses a fresh instance with fixed seed,
  sensor standard deviation 0.05°C and zero process noise.

Full precision remains in the CSVs. Numerical checks cover sign/equilibrium,
independent W/s/J dimensional equivalence, the final input's effect, closed-form
Euler responses including near equilibrium, matched time grids, physical mappings
and fresh-instance noise reproducibility. Each unfinished TODO was also tested
for its own helpful `NotImplementedError`. The last validated solution and
completed-student results are identical.

## Figure placement and review

All ten canonical figures have 220-dpi PNG and text-preserving SVG exports.
The saved 39-slide Lecture 6.1 deck was inspected read-only in PowerPoint.
Temporary overlays used its actual placeholder positions; the deck was not edited.
Legends were moved clear of the time-series and sensitivity traces after review.
Use a **full-width slot for F61-09**: the actual two-state topology and allocation
labels become too small in the current right-hand placeholder. The manifest
allows full-width placement. Keep SVG proportions, reserve the existing footer
and logo area, and avoid reducing label sizes to fit the right-hand box.

## Continuity issues for later work

Tutorial 6.2 remains unchanged. Its present predictor-time split includes a
boundary-crossing target; one-step and rollout scoring use unequal initialisation
conventions, and its bias sign differs from measured-minus-predicted. These need
the separately scoped 6.2 revision. The existing common JSON was fitted on all
28 days by the supplied generator; label it an instructor-supplied common model,
not a held-out validation result. Tutorial 6.1 neither fits nor overwrites it.

Weeks 7–8 notebooks are not present in this checkout. Their supplied local copies
were read for continuity only. Their post-step state labels merit the same timing
review before downstream revision; none was changed in this task. Assessment
materials, Week 4 notebooks, shared input files and existing utility code were
protected by hashes. `week6_day1_outputs/protected_hashes.json` includes the
95 existing protected files and the two unchanged supplied dependency copies.
