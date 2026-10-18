"""Instructor authoring utility for the reviewed Tutorial 4.1a/4.1b pairs.

Not a notebook dependency. Sources below retain the current lecturer edits.
Run from the repository root; execute both solutions afterwards.
"""
from pathlib import Path
import sys
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
pairs = {}
student_cells, solution_cells = [], []

def cell(kind, source, identifier, tags=(), student=None):
    source = source.strip()
    factory = nbf.v4.new_markdown_cell if kind == "markdown" else nbf.v4.new_code_cell
    if kind == "code":
        compile(source, identifier, "exec")
    solution_cells.append(factory(source, id=identifier, metadata={"tags": list(tags)}))
    if "worked-discussion" not in tags:
        student_source = source if student is None else student.strip()
        if kind == "code":
            compile(student_source, identifier+"-student", "exec")
        student_cells.append(factory(student_source, id=identifier, metadata={"tags": list(tags)}))

student_cells, solution_cells = [], []

cell("markdown", r"""
# Tutorial 4.1a: Building demand and predictability

Use **Robin_education_Margarito**, a UCL college laboratory in Building Data Genome 2.
Electricity and hot-water thermal energy are recorded in kWh per hourly interval.
Measured outdoor temperature is used only for descriptive analysis.

Explore meter readings, daily profiles, load-duration curves and demand indicators.
Compare buildings by use, inspect coincident peaks, then fit models and diagnose day-ahead forecasts.

Run from the repository root. Allow **70–90 minutes**: readings and resolution 15,
profiles and indicators 20, aggregation 10, model fitting and forecast discussion 35.
Complete eight short code exercises, including scaling and fitting the three models.
Feature construction, the seasonal benchmark and plotting are supplied.

Training: **8 January–30 September 2017**. Test: **1–28 October 2017**.
Forecasts issued at 23:00 cover the following day's 24 hours.
The small building group uses a complete training week, independently of forecast evaluation.

This notebook follows Lecture 4.1. Section headings identify the accompanying slides.
[Data and preparation](data/week4_day1/robin/README.md). No assessment data are used.
""", "overview", tags=[])

cell("markdown", r"""
### Dataset and references

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. (2020),
[‘The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition’](https://doi.org/10.1038/s41597-020-00712-x),
*Scientific Data*, 7, 368.

[BDG2 dataset and documentation](https://github.com/buds-lab/building-data-genome-project-2)
· [Archived dataset release](https://zenodo.org/records/3887306).
The teaching extracts retain the source's CC BY-SA 4.0 attribution.
""", "data_references", tags=[])

cell("markdown", r"""
## L0. Meter readings and data quality (slides 6, 9 and 12)

The dataset has one row per hour in 2017. We retain the recorded timestamps.
For the forecasting example, assume a reading labelled 23:00 is available at 23:00.

**Predict:** what does an hourly reading of 12 kWh tell you about power?
It corresponds to 12 kW average power over one hour. Over half an hour, the same energy
would correspond to 24 kW. Hourly readings cannot reveal shorter power peaks.

This building has electricity and hot-water thermal-energy meters. Heat delivered
through hot water is an energy quantity, not a water volume or an end-use breakdown.
The two meters are shown separately.

A complete index alone does not establish data quality. Check each meter for missing readings,
zeros and repeated values. Margarito's 2017 electricity is complete
and has no identical-value runs longer than two hours.

**Discuss:** if the heat meter is missing at one timestamp, which analyses remain possible?
What operating information could an external reader infer from these profiles?
""", "l0", tags=[])

cell("code", r"""
from pathlib import Path
import hashlib
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"figure.dpi": 100, "font.size": 12,
                     "axes.grid": True, "grid.alpha": 0.2, "svg.fonttype": "none"})
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    raise FileNotFoundError("Launch Jupyter from the repository root.")
FIGURE_DIR = ROOT / "figures" / "week4_day1" / "robin"
EXPORT_ARTEFACTS = True  # Set False to display plots without exporting files.
MODEL_COLOURS = {"Weekly seasonal naïve": "tab:purple", "Ridge": "tab:blue",
                 "Decision tree": "tab:red", "Random forest": "tab:green"}
SLIDE_NUMBERS = {1: 6, 2: 11, 3: 12, 4: 16, 5: 17, 6: 18, 7: 19, 8: 25,
                 9: 26, 10: 26, 11: 27, 17: 41, 18: 6, 19: 20, 20: 23, 21: 24, 22: 39}
SPEC_SLIDE_NUMBERS = SLIDE_NUMBERS.copy()

NOTEBOOK_NAME = "tutorial_4_1a_building_load_solution.ipynb"
SOURCE_LABEL = "BDG2 Robin_education_Margarito; recorded building timestamps"
DATA_PATH = ROOT / "data" / "week4_day1" / "robin" / "robin_education_margarito_2017.csv"
SOURCE_SHA256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
TRAIN_START, TRAIN_END = "2017-01-08 00:00", "2017-09-30 23:00"
TEST_START, TEST_END = "2017-10-01 00:00", "2017-10-28 23:00"
PLOT_START, PLOT_END = "2017-10-08 00:00", "2017-10-14 23:00"
PROFILES_PATH = ROOT / "data/week4_day1/robin/robin_usage_profiles.csv"
""", "setup", tags=["supplied"])

cell("markdown", r"""
### Supplied plotting helpers

The next two functions keep figure styles and exports consistent. Data preparation,
model fitting and error calculations are shown in the subsequent cells.
""", "plotting_helpers_intro", tags=[])

cell("code", r"""
# Supplied file/plot helper. All feature and model calculations stay in later cells.
def export_figure(fig, number, slug, checkpoint, period, units, description, n):
    label = "BDG2 Robin / UCL; screened building profiles" if number == 19 else SOURCE_LABEL
    if number in [20, 21]:
        label = "BDG2 Robin / UCL; fixed laboratory, residence and office group"
    fig.text(0.01, 0.01, textwrap.fill(
        f"{checkpoint} | {label} | {period} | n={n} | {units}", 118), fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    if EXPORT_ARTEFACTS:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"w4d1_F{number:02d}_{slug}.png"
        fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(FIGURE_DIR / filename.replace(".png", ".svg"), bbox_inches="tight", facecolor="white")
        manifest_path = ROOT / "teaching" / "week4_day1_robin_figure_manifest.csv"
        record = pd.DataFrame([{
            "figure_id": f"F{number:02d}", "filename": f"figures/week4_day1/robin/{filename}",
            "slide_number": SLIDE_NUMBERS[number], "spec_slide_number": SPEC_SLIDE_NUMBERS[number],
            "notebook": NOTEBOOK_NAME, "checkpoint": checkpoint, "data_period": period,
            "units": units, "description": description, "status": "generated",
            "optional": number == 17, "sample_count": n, "source_sha256": (hashlib.sha256(PROFILES_PATH.read_bytes()).hexdigest() if number == 19
                              else hashlib.sha256(GROUP_PATH.read_bytes()).hexdigest() if number in [20, 21]
                              else SOURCE_SHA256),
        }])
        old = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        if not old.empty:
            old = old.loc[old.figure_id != f"F{number:02d}"]
        pd.concat([old, record], ignore_index=True).sort_values("figure_id").to_csv(manifest_path, index=False)
    plt.show()
    plt.close(fig)

""", "figure_export", tags=["supplied"])

cell("code", r"""
def draw_table(ax, table, widths=None, fontsize=11):
    ax.axis("off")
    table = table.copy()
    for column in table.select_dtypes(include="floating"):
        table[column] = table[column].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "undefined")
    artist = ax.table(cellText=table.to_numpy(), colLabels=table.columns,
                      loc="center", cellLoc="center", colWidths=widths)
    artist.auto_set_font_size(False)
    artist.set_fontsize(fontsize)
    artist.scale(1, 1.8)
    for (row, column), cell in artist.get_celld().items():
        cell.set_edgecolor("0.85")
        if row == 0:
            cell.set_facecolor("#24476a")
            cell.set_text_props(color="white", weight="bold")
    return artist
""", "table_plotting", tags=["supplied"])

cell("code", r"""
source = pd.read_csv(DATA_PATH, index_col="timestamp", parse_dates=True).sort_index()
full_index = pd.date_range(source.index.min(), source.index.max(), freq="h", name="timestamp")
missing_timestamps = full_index.difference(source.index)
if not source.index.is_unique or len(missing_timestamps):
    raise ValueError("Forecast lags require unique, complete hourly timestamps.")
data = source.copy()
original_load = data["electricity_kwh"].copy()
original_observed = original_load.notna()
training_original = original_load.loc[TRAIN_START:TRAIN_END]
""", "audit", tags=["supplied"])

cell("markdown", r"""
### Inspect the available meters

The building has electricity and hot-water energy meters. Plot their readings
separately.
""", "meter_exploration-intro", tags=[])

cell("code", r"""
# Meter availability checked against BDG2 metadata and the cleaned meter files.
meter_inventory = pd.DataFrame({
    "Meter": ["Electricity", "Gas", "Hot water", "Chilled water",
              "Steam", "Water", "Irrigation", "Solar"],
    "Series provided": ["Yes", "No", "Yes", "No", "No", "No", "No", "No"],
})
display(meter_inventory)
meter_week = training_original.loc["2017-02-06":"2017-02-12 23:00"]
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
for ax, column, label in zip(axes, ["electricity_kwh", "hotwater_kwh"],
                             ["Electricity", "Hot-water thermal energy"]):
    ax.plot(data.loc[meter_week.index].index, data.loc[meter_week.index, column], color="black")
    ax.set(ylabel="Hourly energy (kWh)", title=label)
axes[1].set_xlabel("Recorded date (Robin / UCL)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
export_figure(fig, 18, "additional_meters", "L0", "6–12 February 2017",
              "electricity and thermal kWh per hourly interval",
              "Separate measured electricity and hot-water energy for the same UCL building", len(meter_week))
""", "meter_exploration", tags=["supplied"])

cell("markdown", r"""
### Summarise data quality

Report missing values, zeros and the longest run of identical electricity readings.
Repeated values over multiple time steps may warrant inspection.
""", "quality_summary-intro", tags=[])

cell("code", r"""
audit = pd.DataFrame({
    "Check": ["Source rows", "Absent timestamps", "Stored electricity NaNs",
              "Stored weather NaNs", "Genuine electricity zeros", "Training target NaNs", "Test target NaNs"],
    "Count": [len(source), len(missing_timestamps), source.electricity_kwh.isna().sum(),
              source.air_temperature_c.isna().sum(), original_load.eq(0).sum(),
              training_original.isna().sum(), original_load.loc[TEST_START:TEST_END].isna().sum()],
})
print(f"{source.index.min()} to {source.index.max()}; nominal interval = {source.index.to_series().diff().mode().iloc[0]}")
audit = pd.concat([audit, pd.DataFrame({
    "Check": ["Stored hot-water NaNs", "Training hot-water NaNs"],
    "Count": [data.hotwater_kwh.isna().sum(), data.loc[TRAIN_START:TRAIN_END, "hotwater_kwh"].isna().sum()],
})], ignore_index=True)
electricity_run_lengths = original_load.groupby(original_load.ne(original_load.shift()).cumsum()).size()
audit.loc[len(audit)] = ["Longest constant electricity run (hours)", int(electricity_run_lengths.max())]
display(audit)
""", "quality_summary", tags=["supplied"])

cell("markdown", r"""
### Read six actual observations

Inspect the timestamps, units and values before calculating daily totals.
""", "meter_rows_plot-intro", tags=[])

cell("code", r"""
meter_rows = data.loc["2017-02-06 08:00":"2017-02-06 13:00",
                      ["electricity_kwh", "hotwater_kwh", "air_temperature_c"]].reset_index()
display(meter_rows)
table = meter_rows.copy()
table["timestamp"] = table.timestamp.dt.strftime("%d %b %Y %H:%M")
table.columns = ["Recorded timestamp", "Electricity (kWh)", "Hot-water energy (kWh)", "Observed temperature (°C)"]
fig, ax = plt.subplots(figsize=(12, 3.3))
ax.set_title("Six observed rows: Robin_education_Margarito", pad=18)
draw_table(ax, table.round(3), widths=[0.28, 0.21, 0.27, 0.24])
export_figure(fig, 1, "meter_rows", "L0", "6 February 2017, 08:00–13:00",
              "hourly kWh; observed °C", "Six actual source rows", 6)
""", "meter_rows_plot", tags=["supplied"])

cell("markdown", r"""
### Check readings before forecasting

The selected 2017 electricity series has no missing readings or long constant
stretches. We use its values directly, without imputation.

There is one missing hot-water reading in training. The supplied plot illustrates
this gap. Hot-water energy and temperature are descriptive measurements, not forecast
inputs. We do not fill missing values in these columns or use them as scored targets.
""", "training_preparation", tags=[])

cell("code", r"""
load_for_features = original_load.loc[:TEST_END].copy()  # No imputation.
training_heat = data.loc[TRAIN_START:TRAIN_END, "hotwater_kwh"]
gap_times = training_heat.index[training_heat.isna()]
gap = gap_times[0]
zoom = pd.date_range(gap - pd.Timedelta(hours=12), gap + pd.Timedelta(hours=12), freq="h")
fig, axes = plt.subplots(3, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1, 1.1]})
for ax, column, label in zip(axes[:2], ["electricity_kwh", "hotwater_kwh"],
                             ["Electricity", "Hot-water thermal energy"]):
    ax.plot(zoom, data.loc[zoom, column], color="black", marker="o", markersize=3)
    ax.axvspan(gap - pd.Timedelta(hours=1), gap + pd.Timedelta(hours=1),
                alpha=0.12, color="tab:red")
    ax.set(title=label, ylabel="Hourly energy (kWh)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
axes[0].annotate("Electricity remains available", (gap, data.loc[gap, "electricity_kwh"]),
                 xytext=(20, 15), textcoords="offset points", fontsize=10)
axes[1].set_xlabel(f"Missing heat reading: {gap:%d %b %Y %H:%M}")
audit_meters = pd.DataFrame({
    "Measurement": ["Electricity", "Hot-water thermal energy", "Outdoor temperature"],
    "Unit": ["kWh / hourly interval", "kWh / hourly interval", "°C observed"],
    "Missing, full 2017": [data[c].isna().sum() for c in ["electricity_kwh", "hotwater_kwh", "air_temperature_c"]],
    "Missing, training": [data.loc[TRAIN_START:TRAIN_END, c].isna().sum() for c in ["electricity_kwh", "hotwater_kwh", "air_temperature_c"]],
})
axes[2].set_title("Counts: full year versus 8 January–30 September 2017", fontsize=12)
draw_table(axes[2], audit_meters, widths=[0.28, 0.28, 0.22, 0.22], fontsize=11)
export_figure(fig, 3, "quality_audit", "L0", f"{gap:%d %B %Y} gap; full year / training counts",
              "hourly kWh; measured °C; missing counts", "Separate meter quality; no filling", len(zoom))
""", "meter_gap_plot", tags=["supplied"])

cell("markdown", r"""
## L1. Hourly readings and daily totals (slide 11)

**Task 1:** sum the hourly electricity readings over each day of 6–12 February 2017.
Use `resample("D").sum(min_count=24)`: incomplete days should have no reported total.

**Predict:** what information is lost when only daily totals are retained?

The two panels use the same dates. The marker identifies the largest hourly reading
and its day in the daily totals. Explain one feature of the hourly trace that the
daily plot conceals.
""", "l1", tags=[])

cell("code", r"""
resolution_hourly = training_original.loc["2017-02-06":"2017-02-12 23:00"]
daily_energy = resolution_hourly.resample("D").sum(min_count=24)
""", "daily_energy", tags=["exercise"], student=r"""
resolution_hourly = training_original.loc["2017-02-06":"2017-02-12 23:00"]
# TODO: resample by day and sum 24 hourly interval readings.
raise NotImplementedError("Construct daily_energy using daily sums, not means.")
""")

cell("code", r"""
peak_time = resolution_hourly.idxmax()
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(resolution_hourly.index, resolution_hourly, color="black")
axes[0].scatter(peak_time, resolution_hourly.loc[peak_time], color="tab:red", zorder=4,
                label=f"Hourly peak: {resolution_hourly.max():.0f} kWh")
axes[0].set(title="Hourly electricity", ylabel="kWh per hourly interval")
axes[0].legend()
axes[1].bar(daily_energy.index + pd.Timedelta(hours=12), daily_energy, width=0.8, color="tab:blue")
axes[1].scatter(peak_time.normalize() + pd.Timedelta(hours=12),
                daily_energy.loc[peak_time.normalize()], color="tab:red", zorder=4,
                label="Day containing the hourly peak")
axes[1].set(title="Daily electricity totals", ylabel="Daily energy (kWh)", xlabel="Recorded date")
axes[1].legend()
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a\n%d %b"))
export_figure(fig, 2, "temporal_resolution", "L1", "6–12 February 2017",
              "hourly interval kWh; daily kWh", "Aligned hourly trace and daily totals; observed peak marked", 168)
""", "resolution_plot", tags=["supplied"])

cell("markdown", r"""
## L2. Daily patterns and measured weather (slides 16–17)

**Task 2:** calculate mean hourly electricity separately for weekdays and weekends.
The supplied counts show how many original observations support each mean.

**Predict:** when will weekday and weekend demand differ most?

Compare the mean profiles with the individual daily traces. Describe the daytime
increase and the overnight demand. What variation does the mean profile conceal?

The temperature plots show measured weather against electricity and thermal energy.
Identify differences between months. Similar temperatures may coincide with different
energy use. These plots alone do not establish a causal relationship.
Future measured temperature is unavailable at a forecast origin and is excluded from the models.
""", "l2", tags=[])

cell("code", r"""
profile = pd.DataFrame({"electricity_kwh": training_original})
profile["hour"] = profile.index.hour
profile["day_type"] = np.where(profile.index.dayofweek < 5, "Weekday", "Weekend")
mean_profiles = profile.groupby(["hour", "day_type"])["electricity_kwh"].mean().unstack()
""", "profiles", tags=["exercise"], student=r"""
profile = pd.DataFrame({"electricity_kwh": training_original})
profile["hour"] = profile.index.hour
profile["day_type"] = np.where(profile.index.dayofweek < 5, "Weekday", "Weekend")
# TODO: group electricity by hour/day_type, take the mean and unstack day_type.
raise NotImplementedError("Construct mean_profiles from original observations.")
""")

cell("code", r"""
profile_counts = profile.groupby(["hour", "day_type"])["electricity_kwh"].count().unstack()
display(profile_counts)
fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
for ax, name, colour in zip(axes, ["Weekday", "Weekend"], ["tab:blue", "tab:orange"]):
    # Grey lines: individual observed days from the same period as the mean.
    for day, readings in training_original.groupby(training_original.index.normalize()):
        day_type = "Weekday" if day.dayofweek < 5 else "Weekend"
        if day_type == name:
            ax.plot(readings.index.hour, readings, color="0.75", linewidth=0.6, alpha=0.3)
    counts = profile_counts[name]
    ax.plot(mean_profiles.index, mean_profiles[name], marker="o", markersize=4, color=colour,
            linewidth=2, label=f"Training mean")
    ax.set(title=name, xlabel="Hour of day", ylabel="Hourly energy (kWh)", xticks=range(0, 24, 3))
    ax.legend(fontsize=9)
fig.suptitle("Coloured: training mean; grey: individual observed training days")
export_figure(fig, 4, "weekday_weekend", "L2", "8 January–30 September 2017",
              "kWh/hourly interval", "Observed day-type means and individual daily variation; no imputation",
              int(training_original.notna().sum()))
""", "profile_plot", tags=["supplied"])

cell("code", r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
pair_counts = []
for ax, column, title in zip(axes, ["electricity_kwh", "hotwater_kwh"],
                             ["Electricity", "Hot-water thermal energy"]):
    pairs = data.loc[TRAIN_START:TRAIN_END, ["air_temperature_c", column]].dropna()
    pair_counts.append(len(pairs))
    print(title, ": retained", len(pairs), "; excluded", len(training_original) - len(pairs))
    points = ax.scatter(pairs.air_temperature_c, pairs[column], c=pairs.index.month,
                        cmap="viridis", vmin=1, vmax=9, s=8, alpha=0.35, rasterized=True)
    ax.set(title=title, xlabel="Observed outdoor temperature (°C)", ylabel="Hourly energy (kWh)")
fig.colorbar(points, ax=axes[1], label="Month", ticks=range(1, 10))
export_figure(fig, 5, "temperature_relationship", "L2", "8 January–30 September 2017",
              "observed °C; hourly kWh", f"Observed pairs only: electricity n={pair_counts[0]}, heat n={pair_counts[1]}", pair_counts[0])
""", "temperature_plot", tags=["supplied"])

cell("markdown", r"""
## L3. Load-duration curve and demand indicators (slides 18–19)

**Task 3:** sort the observed training electricity from largest to smallest.
Convert interval energy to hourly average power by dividing by the one-hour duration.
The numerical values are equal, but the units differ.

**Predict:** can a sorted curve tell us when a peak happened?

The horizontal axis is the percentage of observed hours exceeded.
The 90th percentile is exceeded during approximately 10% of hours.
Compare the peak, median and sustained demand. Sorting removes the chronological order.
""", "duration_intro", tags=[])

cell("code", r"""
hourly_power_kw = training_original / 1.0  # kWh divided by a one-hour interval.
duration_kw = hourly_power_kw.sort_values(ascending=False)
exceedance_percent = 100 * (np.arange(len(duration_kw)) + 0.5) / len(duration_kw)
""", "duration", tags=["exercise"], student=r"""
hourly_power_kw = training_original / 1.0  # kWh divided by a one-hour interval.
# TODO: sort hourly_power_kw in descending order, then calculate percentage positions.
raise NotImplementedError("Construct duration_kw and exceedance_percent.")
""")

cell("code", r"""
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(exceedance_percent, duration_kw.to_numpy(), color="black")
for percentile, position, colour in [(1.0, 0, "tab:red"), (0.9, 10, "tab:orange"), (0.5, 50, "tab:blue")]:
    value = hourly_power_kw.quantile(percentile)
    label = "Maximum" if percentile == 1 else f"{int(100 * percentile)}th percentile"
    ax.scatter(position, value, color=colour, zorder=4)
    ax.annotate(f"{label}: {value:.0f} kW", (position, value),
                xytext=(20, -18 if percentile == 1 else 10), textcoords="offset points", color=colour)
ax.set(title="Margarito: load-duration curve", xlabel="Observed hours exceeded (%)",
       ylabel="Hourly average electricity demand (kW)", xlim=(0, 100))
export_figure(fig, 6, "load_duration", "L3", "8 January–30 September 2017",
              "hourly average kW", "Observed training duration curve; maximum, median and 90th percentile", len(hourly_power_kw))
""", "duration_plot", tags=["supplied"])

cell("markdown", r"""
**Task 4:** calculate load factor and overnight energy share.

Load factor is mean hourly power divided by maximum hourly power.
Overnight share is the fraction of total energy consumed at 00:00–05:59.
The supplied code adds the weekend/weekday mean-demand ratio and peak timestamp.

Interpret two indicators alongside the profiles and duration curve. A high overnight
share alone does not establish waste: laboratory equipment or other services may operate continuously.
""", "indicators_intro", tags=[])

cell("code", r"""
load_factor = hourly_power_kw.mean() / hourly_power_kw.max()
overnight = training_original.index.hour < 6
overnight_share = training_original.loc[overnight].sum() / training_original.sum()
""", "indicators", tags=["exercise"], student=r"""
overnight = training_original.index.hour < 6
# TODO: calculate load_factor and overnight_share using the definitions above.
raise NotImplementedError("Calculate the two demand indicators.")
""")

cell("code", r"""
weekend = training_original.index.dayofweek >= 5
weekend_ratio = training_original.loc[weekend].mean() / training_original.loc[~weekend].mean()
demand_indicators = pd.DataFrame({
    "Indicator": ["Load factor", "Overnight energy share", "Weekend/weekday ratio", "Peak timing"],
    "Value": [f"{load_factor:.3f}", f"{100 * overnight_share:.1f}%", f"{weekend_ratio:.3f}",
              training_original.idxmax().strftime("%d %b %Y %H:%M")],
    "Definition": ["Mean / maximum hourly power", "00:00–05:59 energy / total energy",
                   "Mean weekend / mean weekday demand", "Maximum hourly average demand"],
})
display(demand_indicators)
fig, ax = plt.subplots(figsize=(12, 3.4))
ax.set_title("Margarito: summary indicators over the same training period", pad=18)
draw_table(ax, demand_indicators, widths=[0.25, 0.25, 0.5], fontsize=11)
export_figure(fig, 7, "demand_indicators", "L3", "8 January–30 September 2017",
              "dimensionless ratios; energy share %; recorded timestamp",
              "Load factor, overnight share, weekend ratio and peak timing", len(training_original))
""", "indicators_plot", tags=["supplied"])

cell("markdown", r"""
### Compare building uses (slide 20)

**Predict:** which usage type will show the largest weekday/weekend difference?

Compare the Robin-site residential, education and office buildings retained after
data-quality screening. The count table reports the number in each category.
The supplied table reports each building's mean electricity at each hour, separately
for weekdays and weekends, using only observed readings from the training period.
Source NaNs and exact constant stretches of at least 24 hours are excluded from
these descriptive means. Such repeats may reflect earlier gap filling or a meter
problem, but the cause is not known. The supplied table reports exclusions.
Screening uses training data only; no source readings are overwritten or filled.

To compare shapes rather than building sizes, divide each building's readings by its
maximum retained training reading. Grey lines show individual buildings. The black
line gives each building equal weight, regardless of floor area or number of readings.
This is a normalised profile, not a comparison of absolute energy use or efficiency.

After plotting, identify one weekday/weekend difference and one source of variation
within a usage type. Explain why the group mean alone is insufficient for forecasting
an individual building.
""", "usage_type_intro", tags=[])

cell("code", r"""
usage_profiles = pd.read_csv(PROFILES_PATH)
profile_counts = usage_profiles.groupby("usage_type").building_id.nunique()
display(profile_counts.rename("Buildings"))
profile_quality = pd.read_csv(ROOT / "data/week4_day1/robin/robin_profile_quality.csv")
display(profile_quality.groupby("usage_type")[["source_missing", "flagged_flat_hours", "retained_hours"]].sum())
categories = ["Lodging/residential", "Education", "Office"]
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey=True)
for row, day_type in enumerate(["Weekday", "Weekend"]):
    for column, usage_type in enumerate(categories):
        ax = axes[row, column]
        subset = usage_profiles.loc[usage_profiles.usage_type.eq(usage_type)
                                    & usage_profiles.day_type.eq(day_type)]
        for building_id, group in subset.groupby("building_id"):
            group = group.sort_values("hour")
            ax.plot(group.hour, group.normalised_mean, color="0.8", linewidth=1, alpha=0.8)
        average = subset.groupby("hour").normalised_mean.mean()
        ax.plot(average.index, average, color="black", linewidth=2.5, label="Mean across buildings")
        ax.set(title=f"{usage_type}: {day_type.lower()} (n={profile_counts[usage_type]})",
               xlim=(0, 23), ylim=(0, 1), xticks=[0, 6, 12, 18, 23], xlabel="Hour of day")
        if column == 0:
            ax.set_ylabel("Normalised hourly electricity")
axes[0, 0].legend(fontsize=9, loc="upper left")
fig.suptitle("Robin / UCL: daily profiles by building usage and weekday/weekend", fontsize=16)
export_figure(fig, 19, "usage_type_profiles", "L3", "8 January–30 September 2017",
              "dimensionless; each building's training maximum = 1",
              "Retained individual-building profiles (grey) and equal-weight usage means (black)", int(usage_profiles.building_id.nunique()))
usage_means = usage_profiles.groupby(["usage_type", "day_type", "hour"]).normalised_mean.mean()
display(usage_means.groupby(["usage_type", "day_type"]).max().unstack().round(3))
""", "usage_type_plot", tags=["supplied"])

cell("code", r"""
print("Education and office profiles have stronger weekday daytime peaks; residential profiles peak later.")
print("Buildings within each group differ in their baseline and peak levels. A group average cannot replace a building-specific forecast.")
""", "usage_type_discussion", tags=["worked-discussion"])

cell("markdown", r"""
## L4. Aggregation and coincident peaks (slides 23–24)

Use a fixed group: the Margarito laboratory, Armand residence and Sammie office.
They were selected from metadata and training-data quality, before inspecting aggregate peaks.
Readings are complete over **6–12 February 2017**. No values are filled or replaced with zero.

**Predict:** will the peak of their sum equal the sum of their separate peaks?

The supplied calculation sums original, simultaneous readings. Do not sum normalised mean profiles
to represent measured demand. The hourly energy divided by one hour gives average kW.
""", "aggregation_intro", tags=[])

cell("code", r"""
GROUP_PATH = ROOT / "data/week4_day1/robin/robin_group_electricity_2017_week.csv"
group_energy = pd.read_csv(GROUP_PATH, index_col="timestamp", parse_dates=True)
group_power = group_energy / 1.0  # One-hour intervals: kWh / h = kW.
aggregate_power = group_power.sum(axis=1, min_count=group_power.shape[1])
individual_peaks = group_power.max()
sum_individual_peaks = individual_peaks.sum()
aggregate_peak = aggregate_power.max()
aggregate_peak_time = aggregate_power.idxmax()
group_peak_table = pd.DataFrame({"Peak (kW)": individual_peaks,
                                "Peak time": group_power.idxmax()})
display(group_peak_table)
print(f"Sum of separate peaks = {sum_individual_peaks:.3f} kW")
print(f"Peak of simultaneous sum = {aggregate_peak:.3f} kW at {aggregate_peak_time}")
print(f"Group energy over the week = {group_energy.sum().sum():.3f} kWh")

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
for building_id in group_power:
    axes[0].plot(group_power.index, group_power[building_id],
                 label=building_id.replace("Robin_", "").replace("_", " / "))
axes[0].set(title="Three Robin buildings: original aligned readings", ylabel="Hourly average power (kW)")
axes[0].legend(fontsize=10)
axes[1].plot(aggregate_power.index, aggregate_power, color="black", label="Simultaneous sum")
axes[1].scatter(aggregate_peak_time, aggregate_peak, color="tab:red", zorder=4,
                label=f"Aggregate peak: {aggregate_peak:.0f} kW")
axes[1].set(ylabel="Group average power (kW)", xlabel="Recorded date")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
axes[1].legend()
export_figure(fig, 20, "group_aggregation", "L4", "6–12 February 2017",
              "hourly average kW", "Fixed three-building group; not a documented feeder", len(group_power))
""", "aggregation", tags=["supplied"])

cell("code", r"""
fig, ax = plt.subplots(figsize=(10, 4))
values = [sum_individual_peaks, aggregate_peak]
bars = ax.bar(["Sum of individual peaks", "Peak of simultaneous sum"], values,
               color=["0.6", "tab:blue"], width=0.55)
ax.bar_label(bars, fmt="%.1f kW", padding=5)
ax.set(ylabel="Hourly average power (kW)", ylim=(0, max(values) * 1.18),
       title="Peak coincidence over identical timestamps")
export_figure(fig, 21, "peak_coincidence", "L4", "6–12 February 2017",
              "hourly average kW", "Noncoincident peaks versus aggregate peak; energy unchanged", len(group_power))
""", "coincidence_plot", tags=["supplied"])

cell("markdown", r"""
**Interpret:** use the peak timestamps to explain the difference between the bars.
Does aggregation reduce the group's total energy consumption?
Annotate one demand plot and this peak comparison, then explain one result from each.
The individual and aggregate curves describe the same energy, grouped differently.
""", "aggregation_question", tags=[])

cell("markdown", r"""
## L5. Construct day-ahead forecasts (slide 25)

At 23:00, predict all 24 hourly electricity readings of the following day.
The first target is one hour ahead and the last is 24 hours ahead.

Compare the **weekly seasonal benchmark**, **Ridge regression**, a **decision tree**
and a **random forest**. The benchmark copies the same hour of the previous week.
The trained models combine daily/weekly load lags with target-calendar variables.

Feature preparation and the seasonal benchmark are supplied. Complete one scaling
exercise, then fit Ridge, the tree and forest in separate cells. Fit on the earlier
training data only. Earlier test observations may supply later lag inputs, but each trained
model is unchanged during evaluation.

**Check:** why is observed lag 1 available for only the first target at a common
23:00 origin, whereas lags 24 and 168 are available for every next-day target?
""", "l3", tags=[])

cell("markdown", r"""
### Supplied predictors

Construct lags on the full hourly grid before selecting rows, so `shift(24)`
means 24 hours earlier. Ridge uses lag 24, lag 168, hour sine/cosine,
weekday sine/cosine and a weekend indicator. Calendar values describe the target
and are known in advance. Sine/cosine periods are 24 hours and seven days.

The tree and forest both use lag 24, lag 168, integer hour, integer weekday and
weekend indicator. Inspect the two calendar encodings below before fitting.
""", "features_description", tags=[])

cell("code", r"""
features = pd.DataFrame(index=load_for_features.index)
features["load_lag_24"] = load_for_features.shift(24)
features["load_lag_168"] = load_for_features.shift(168)
features["hour"] = features.index.hour
features["day_of_week"] = features.index.dayofweek
features["is_weekend"] = (features.index.dayofweek >= 5).astype(int)
features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
features["day_sin"] = np.sin(2 * np.pi * features["day_of_week"] / 7)
features["day_cos"] = np.cos(2 * np.pi * features["day_of_week"] / 7)
""", "features", tags=["supplied"])

cell("markdown", r"""
### Calendar variables: integers or cyclical features?

Hour of the day can be represented by an integer from 0 to 23. Weekday runs from Monday=0 to
Sunday=6. These values are known for every forecast target.

For a linear model, an integer hour contributes $\beta_h h$, which imposes a straight
trend across the day and a jump between 23:00 and 00:00. Cyclical features use
$h_{\sin}=\sin(2\pi h/24)$ and $h_{\cos}=\cos(2\pi h/24)$.
They represent neighbouring hours on either side of midnight as nearby points. Use the same construction
with period 7 for weekday.

**Predict:** why do we need both sine and cosine rather than sine alone?

Ridge uses the sine/cosine pairs. The decision tree and random forest use the **same
integer calendar features**. Trees learn threshold rules, such as `hour <= 8`,
without imposing a straight trend. They can represent nonlinear schedules through
several splits, although integer encoding does not itself make the clock circular.

These feature choices are fixed before fitting. We are illustrating encodings,
not selecting one using final test scores.
[Further reading: scikit-learn's calendar-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html).
""", "calendar_encoding", tags=[])

cell("code", r"""
selected_targets = pd.DatetimeIndex(["2017-09-29 23:00", "2017-09-30 00:00", "2017-09-30 12:00"])
calendar_rows = features.loc[selected_targets,
    ["load_lag_24", "load_lag_168", "hour", "day_of_week", "hour_sin", "hour_cos", "day_sin", "day_cos"]]
display(calendar_rows.round(3))

hours = np.arange(24)
angle = 2 * np.pi * hours / 24
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(hours, np.zeros(24), color="0.7", s=30)
axes[0].set(title="Integer hour: a gap at midnight",
            xlabel="Hour", yticks=[], xticks=[0, 6, 12, 18, 23], ylim=(-0.5, 0.5))
axes[1].plot(np.cos(angle), np.sin(angle), color="0.7")
axes[1].scatter(np.cos(angle), np.sin(angle), color="0.7", s=30)
for hour, colour in [(23, "tab:orange"), (0, "tab:blue")]:
    axes[0].scatter(hour, 0, color=colour, s=85, zorder=4)
    axes[0].annotate(f"{hour}:00", (hour, 0), xytext=(0, 20),
                     textcoords="offset points", ha="center", color=colour)
    axes[1].scatter(np.cos(angle[hour]), np.sin(angle[hour]), color=colour, s=85, zorder=4)
    axes[1].annotate(f"{hour}:00", (np.cos(angle[hour]), np.sin(angle[hour])),
                     xytext=(-65, 10 if hour == 0 else -25), textcoords="offset points", color=colour)
axes[1].set(title="Cyclical hour: neighbours across midnight",
            xlabel="hour_cos", ylabel="hour_sin", aspect="equal", xlim=(-1.3, 1.3), ylim=(-1.3, 1.3))
export_figure(fig, 22, "calendar_encodings", "L5", "Calendar encoding; actual feature rows: 29–30 September 2017",
              "dimensionless calendar features", "Integer versus sine/cosine hour; target-time examples printed above", len(selected_targets))
""", "calendar_plot", tags=["supplied"])

cell("markdown", r"""
### Chronological split and training-only scaling

Keep 1–7 January as lag history. Fit on 8 January–30 September 2017 and evaluate
on 1–28 October 2017. The selected periods contain complete electricity readings and available
predictors: 6,384 training and 672 test targets. No target is filled or excluded.

Fit `StandardScaler` on training predictors only. Reuse it to transform test inputs.
The benchmark uses unscaled electricity readings. Both tree-based models use unscaled predictors.
""", "l4", tags=[])

cell("code", r"""
RIDGE_COLUMNS = ["load_lag_24", "load_lag_168", "hour_sin", "hour_cos",
                 "day_sin", "day_cos", "is_weekend"]
TREE_COLUMNS = ["load_lag_24", "load_lag_168", "hour", "day_of_week", "is_weekend"]

X_train = features.loc[TRAIN_START:TRAIN_END].copy()
X_test = features.loc[TEST_START:TEST_END].copy()
y_train = original_load.loc[X_train.index]
y_test = original_load.loc[X_test.index]
test_origins = X_test.index.normalize() - pd.Timedelta(hours=1)
forecasts = pd.DataFrame(index=y_test.index)
print("Training / test targets:", len(y_train), len(y_test))
""", "split", tags=["supplied"])

cell("markdown", r"""
### Task 5: scale the Ridge inputs

Create `ridge_train_scaled` and `ridge_test_scaled` using the supplied scaler.
Use `fit_transform` on `X_train[RIDGE_COLUMNS]`, then `transform` on
`X_test[RIDGE_COLUMNS]`. Do not fit again on test data.

The returned arrays retain the row order and seven input columns. The seasonal
benchmark and both tree-based models use unscaled inputs.
""", "scale_description", tags=[])

cell("code", r"""
scaler = StandardScaler()
ridge_train_scaled = scaler.fit_transform(X_train[RIDGE_COLUMNS])
ridge_test_scaled = scaler.transform(X_test[RIDGE_COLUMNS])
""", "scale", tags=["exercise"], student=r"""
scaler = StandardScaler()
# TODO: fit the scaler on X_train[RIDGE_COLUMNS] and create ridge_train_scaled.
# Use the SAME scaler to create ridge_test_scaled from X_test[RIDGE_COLUMNS].
# Hint: fit_transform for training; transform for test.
raise NotImplementedError("Scale the Ridge inputs using training data only.")
""")

cell("markdown", r"""
### Weekly seasonal-naïve benchmark

For each target $t$, use $\hat y_t=y_{t-168}$. The predictor is observed by the
previous day's 23:00 origin. Keep forecasts nonnegative.
""", "l5_baselines", tags=[])

cell("code", r"""
forecasts["Weekly seasonal naïve"] = X_test["load_lag_168"].clip(lower=0)
""", "weekly_naive", tags=["supplied"])

cell("markdown", r"""
### Task 6: fit Ridge regression

For scaled inputs $z_{t,j}$, predict $\hat y_t=\beta_0+\sum_j\beta_j z_{t,j}$.
The intercept $\beta_0$ is in kWh. Each $\beta_j$ weights one scaled predictor.
Ridge penalises large coefficients through $\alpha\sum_j\beta_j^2$.
We fix $\alpha=10$ before evaluating the test period. In practice, the hyperparameter $\alpha$ would typically be tuned on a separate validation set.

Fit on the scaled training inputs and observed training targets. Predict the scaled
test inputs, then attach the original test timestamps using a pandas Series.
Store the fitted model in `ridge_model` and the indexed forecasts in `ridge_raw`.
""", "ridge_description", tags=[])

cell("code", r"""
ridge_model = Ridge(alpha=10.0)
ridge_model.fit(ridge_train_scaled, y_train)
ridge_raw = pd.Series(ridge_model.predict(ridge_test_scaled), index=y_test.index)
forecasts["Ridge"] = ridge_raw.clip(lower=0)
""", "ridge_fit", tags=["exercise"], student=r"""
ridge_model = Ridge(alpha=10.0)
# TODO: fit ridge_model using ridge_train_scaled and y_train.
# Predict ridge_test_scaled and create ridge_raw as a Series indexed by y_test.index.
raise NotImplementedError("Fit Ridge on training data and predict the aligned test targets.")

# Supplied nonnegative correction.
forecasts["Ridge"] = ridge_raw.clip(lower=0)
""")

cell("markdown", r"""
### Task 7: fit a decision tree

A decision tree partitions the feature space using threshold rules and predicts the
mean training load in each final region. It can learn nonlinear relationships between
load lags and operating schedules.

Use `TREE_COLUMNS`: lag 24, lag 168, integer hour, integer weekday and weekend indicator.
No scaling is needed. Fix maximum depth at 6 and at least four training cases per leaf.
These settings limit complexity and are specified before test evaluation.

**Predict:** will a single tree give smoother forecasts than Ridge?
Inspect its trace in the forecast comparison below.

Fit on `X_train[TREE_COLUMNS]` and `y_train`, then predict `X_test[TREE_COLUMNS]`.
Store the model in `tree_model` and indexed forecasts in `tree_raw`.
The nonnegative correction is supplied.
""", "tree_description", tags=[])

cell("code", r"""
tree_model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42)
tree_model.fit(X_train[TREE_COLUMNS], y_train)
tree_raw = pd.Series(tree_model.predict(X_test[TREE_COLUMNS]), index=y_test.index)
forecasts["Decision tree"] = tree_raw.clip(lower=0)
""", "tree_fit", tags=["exercise"], student=r"""
tree_model = DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42)
# TODO: fit tree_model using X_train[TREE_COLUMNS] and y_train.
# Predict X_test[TREE_COLUMNS]; store tree_raw as a Series indexed by y_test.index.
# The tree uses unscaled predictors.
raise NotImplementedError("Fit the decision tree and create aligned raw test predictions.")

forecasts["Decision tree"] = tree_raw.clip(lower=0)
""")

cell("markdown", r"""
### Task 8: fit a random forest

A random forest averages predictions from many trees, fitted using resampled training
rows and random subsets of features at each split. Averaging can reduce the instability
of a single tree.

Use exactly the same five integer/load inputs as the decision tree, without scaling.
Fit once, then predict the same test targets. The settings below are fixed in advance.

**Predict:** how might averaging trees change the forecast trace and its errors?
Compare all methods using the same actual observations.

Fit `forest_model` on the same inputs and targets as the decision tree.
Store indexed predictions in `forest_raw`. Keep the supplied settings unchanged.
""", "forest_description", tags=[])

cell("code", r"""
forest_model = RandomForestRegressor(n_estimators=150, min_samples_leaf=4,
                                     max_features=0.8, random_state=42, n_jobs=1)
forest_model.fit(X_train[TREE_COLUMNS], y_train)
forest_raw = pd.Series(forest_model.predict(X_test[TREE_COLUMNS]), index=y_test.index)
forecasts["Random forest"] = forest_raw.clip(lower=0)
""", "forest_fit", tags=["exercise"], student=r"""
forest_model = RandomForestRegressor(n_estimators=150, min_samples_leaf=4,
                                     max_features=0.8, random_state=42, n_jobs=1)
# TODO: fit forest_model on the SAME inputs/targets as the decision tree.
# Predict X_test[TREE_COLUMNS]; store forest_raw as a Series indexed by y_test.index.
raise NotImplementedError("Fit the random forest and create aligned raw test predictions.")

forecasts["Random forest"] = forest_raw.clip(lower=0)
""")

cell("code", r"""
forecast_contract = pd.DataFrame({
    "Item": ["Forecast issue", "Targets per issue", "Inputs", "Training", "Test"],
    "Setting": ["23:00 on the preceding day", "Next day 00:00–23:00 (leads 1–24)",
                "Observed lags 24/168 and known target calendar",
                f"{TRAIN_START} to {TRAIN_END}; n={len(y_train):,}",
                f"{TEST_START} to {TEST_END}; n={len(y_test):,}"],
})
fig, ax = plt.subplots(figsize=(12, 3.8))
ax.set_title("Day-ahead load forecasting: train once, predict each next day", pad=18)
draw_table(ax, forecast_contract, widths=[0.25, 0.75], fontsize=11)
export_figure(fig, 8, "load_forecast_contract", "L5", "2017 training / October test",
              "recorded timestamps; kWh targets", "Daily origin, target hours, inputs and chronological split", len(y_test))
""", "forecast_contract", tags=["supplied"])

cell("markdown", r"""
## L6. Regular patterns and forecast errors (slide 26)

Reuse the supplied metric functions from Tutorial 1.2:
$e_t=y_t-\hat y_t$, $\mathrm{MAE}=\mathrm{mean}(|e_t|)$ and
$\mathrm{RMSE}=\sqrt{\mathrm{mean}(e_t^2)}$.
Both metrics are in kWh. The functions check timestamps and finite values.

**Predict:** which patterns will last week's profile reproduce, and when might the fitted models improve forecasts?

The table uses all 28 test days. The plot shows **8–14 October 2017**.
Its shaded days are selected after evaluation solely for diagnosis: they have
the smallest and largest daily Ridge MAE within this displayed week.

Identify an error in level, peak magnitude or timing. What additional information
would help explain it?
""", "l6", tags=[])

cell("code", r"""
def metric_arrays(actual, forecast):
    '''Check paired Series before converting them to numerical arrays.'''
    if not isinstance(actual, pd.Series) or not isinstance(forecast, pd.Series):
        raise TypeError("Pass actual and forecast as pandas Series.")
    if len(actual) == 0 or len(actual) != len(forecast) or not actual.index.equals(forecast.index):
        raise ValueError("Actual and forecast must have equal, non-empty aligned timestamps.")
    if not actual.index.is_unique:
        raise ValueError("Metric timestamps must be unique.")
    a, f = actual.to_numpy(dtype=float), forecast.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(f).all():
        raise ValueError("Resolve/count non-finite inputs upstream; metrics never drop pairs.")
    return a, f

def mae(actual, forecast):
    a, f = metric_arrays(actual, forecast)
    return float(np.mean(np.abs(a - f)))

""", "metric_checks_and_mae_mape", tags=["supplied"])

cell("code", r"""
def rmse(actual, forecast):
    a, f = metric_arrays(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2)))
""", "rmse", tags=["supplied"])

cell("code", r"""
rows = []
for name in forecasts:
    rows.append({"Model": name, "MAE (kWh)": mae(y_test, forecasts[name]),
                 "RMSE (kWh)": rmse(y_test, forecasts[name]), "n": len(y_test)})
comparison = pd.DataFrame(rows).set_index("Model")
display(comparison.round(3))
""", "comparison", tags=["supplied"])

cell("code", r"""
plot_index = y_test.loc[PLOT_START:PLOT_END].index
daily_ridge_mae = (y_test.loc[plot_index] - forecasts.loc[plot_index, "Ridge"]).abs().resample("D").mean()
diagnostic_days = [daily_ridge_mae.idxmin(), daily_ridge_mae.idxmax()]
print("Diagnostic days (smallest/largest Ridge MAE in display week):",
      [day.strftime("%d %b %Y") for day in diagnostic_days])
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(plot_index, y_test.loc[plot_index], color="black", linewidth=2, label="Actual")
for name in forecasts:
    ax.plot(plot_index, forecasts.loc[plot_index, name], color=MODEL_COLOURS[name], label=name)
for day, label, colour in zip(diagnostic_days, ["Smallest daily Ridge MAE", "Largest daily Ridge MAE"],
                              ["tab:green", "tab:red"]):
    ax.axvspan(day, day + pd.Timedelta(days=1), color=colour, alpha=0.10, label=label)
ax.set(title="Day-ahead forecasts: weekly benchmark, Ridge, tree and forest",
       xlabel="Target date (recorded building timestamps)", ylabel="Hourly energy (kWh)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(ncol=3, fontsize=9)
export_figure(fig, 9, "load_forecasts", "L6", "8–14 October 2017",
              "kWh/hourly interval", "Full week; two diagnostic days selected by daily Ridge MAE, not for tuning", len(plot_index))
""", "forecast_plot", tags=["supplied"])

cell("code", r"""
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_title("Full test period: 1–28 October 2017", pad=18)
draw_table(ax, comparison.reset_index(), widths=[0.4, 0.23, 0.23, 0.14], fontsize=12)
export_figure(fig, 10, "load_scores", "L6", "1–28 October 2017",
              "MAE/RMSE kWh", "Weekly benchmark, Ridge, decision tree and random forest on common test targets", len(y_test))
""", "score_table_plot", tags=["supplied"])

cell("markdown", r"""
## L7. Average errors and daily peaks (slide 27)

A low average error can coexist with a missed daily peak.
Depending on the application, error timing may be more important than error magnitude.
The two panels show the same illustrative days identified above.

For each model, compare daily MAE with the absolute peak-magnitude error:
$|\max_{t\in d}\hat P_t-\max_{t\in d}P_t|$.
Here $P_t$ is average power over the hour, numerically equal to interval kWh divided by one hour.
Peak magnitude and timing are separate. Markers show when each daily maximum occurs.

**Predict:** will the model with lower daily MAE always have the smaller peak error?

Inspect the markers and scores. Would the preferred forecast necessarily be the same
for estimating daily consumption and anticipating the largest hourly demand?
""", "l7", tags=[])

cell("code", r"""
daily_rows = []
fig = plt.figure(figsize=(14, 8))
grid = fig.add_gridspec(2, 2, height_ratios=[2.4, 1.2])
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
score_axis = fig.add_subplot(grid[1, :])

for ax, day in zip(axes, diagnostic_days):
    index = y_test.loc[day:day + pd.Timedelta(hours=23)].index
    actual_power = y_test.loc[index] / 1.0
    ax.plot(index, actual_power, color="black", linewidth=2, label="Actual")
    ax.scatter(actual_power.idxmax(), actual_power.max(), color="black", marker="*", s=130, zorder=5)
    for name in forecasts:
        prediction_power = forecasts.loc[index, name] / 1.0
        ax.plot(index, prediction_power, color=MODEL_COLOURS[name], label=name)
        ax.scatter(prediction_power.idxmax(), prediction_power.max(), color=MODEL_COLOURS[name],
                    marker="o", s=60, zorder=4)
        daily_rows.append({"day": day, "model": name,
                           "mae_kwh": mae(y_test.loc[index], forecasts.loc[index, name]),
                           "peak_error_kw": abs(prediction_power.max() - actual_power.max()),
                           "actual_peak_time": actual_power.idxmax(),
                           "forecast_peak_time": prediction_power.idxmax()})
    ax.set(title=day.strftime("%d %B %Y"),
           xlabel="Target hour", ylabel="Hourly average power (kW)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=9)
daily_peak_scores = pd.DataFrame(daily_rows)

# Keep the numerical labels below the plots rather than over the curves.
peak_table = pd.DataFrame(index=forecasts.columns.copy())
for day in diagnostic_days:
    scores = daily_peak_scores.loc[daily_peak_scores.day.eq(day)].set_index("model")
    peak_table[f"{day:%d %b} MAE (kWh)"] = scores.mae_kwh
    peak_table[f"{day:%d %b} peak error (kW)"] = scores.peak_error_kw
peak_table.index.name = "Model"
draw_table(score_axis, peak_table.reset_index(), widths=[0.28, 0.18, 0.18, 0.18, 0.18], fontsize=11)
export_figure(fig, 11, "daily_peak_errors", "L7",
              ", ".join(day.strftime("%d %B 2017") for day in diagnostic_days),
              "MAE kWh; hourly peak kW", "Four models on two fixed diagnostic days; peak timing and magnitude", 48)
""", "peak_diagnostics", tags=["supplied"])

cell("markdown", r"""
## L8. Evidence-based interpretation (slide 28)

Write three or four sentences. Identify one recurring demand pattern using a plot
and an indicator. Explain one aggregation result using peak timing. Compare the four
forecasts with numerical evidence. State one measurement limitation and what additional
evidence would support an operational recommendation.
""", "l8", tags=[])

cell("code", r"""
print(f"Load factor is {load_factor:.3f}; {100 * overnight_share:.1f}% of training energy occurs at 00:00–05:59.")
print(f"The aggregate peak is {aggregate_peak:.1f} kW versus {sum_individual_peaks:.1f} kW for separate peaks. Peak timing differs; total energy is unchanged.")
baseline = comparison.loc["Weekly seasonal naïve", "RMSE (kWh)"]
improvement = 100 * (baseline - comparison.loc["Ridge", "RMSE (kWh)"]) / baseline
print(f"Ridge RMSE is {comparison.loc['Ridge', 'RMSE (kWh)']:.3f} kWh, {improvement:.1f}% below the weekly benchmark in this test period.")
print("Hourly means hide shorter peaks. The plots do not establish occupancy or causes; operating schedules and equipment records would help explain demand.")
for metric in ["MAE (kWh)", "RMSE (kWh)"]:
    best = comparison[metric].idxmin()
    print(f"Lowest {metric}: {best}, {comparison.loc[best, metric]:.3f}. Rankings depend on the metric and this fixed test period.")
print("Ridge uses paired sine/cosine features, which avoid a straight hour trend and the midnight discontinuity. Sine alone cannot distinguish some hours, such as midnight and midday.")
print("The tree and forest use identical integer features. Multiple threshold splits can represent nonlinear schedules; averaging trees can reduce the instability of a single tree.")
print("On 13 October, Ridge predicts the peak two hours late. Compare all four peak markers and magnitude errors: low average error does not guarantee correct timing.")
first_day = daily_peak_scores.loc[daily_peak_scores.day.eq(diagnostic_days[0])].set_index("model")
print(f"On {diagnostic_days[0]:%d October}, {first_day.mae_kwh.idxmin()} has the lowest daily MAE, but {first_day.peak_error_kw.idxmin()} has the smallest peak-magnitude error.")
""", "worked_discussion", tags=["worked-discussion"])

cell("code", r"""
# Optional instructor exports; these files are not required to run the notebook.
if EXPORT_ARTEFACTS:
    result_dir = ROOT / "teaching/week4_day1_outputs/robin"
    result_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(result_dir / "load_forecasts.csv")
    comparison.to_csv(result_dir / "load_scores.csv")
    daily_peak_scores.to_csv(result_dir / "load_daily_peak_scores.csv", index=False)
""", "export_results", tags=["supplied"])

pairs["tutorial_4_1a_building_load"] = (student_cells, solution_cells)

student_cells, solution_cells = [], []

cell("markdown", r"""
# Tutorial 4.1b: One-hour-ahead PV point forecasting

Use **GEFCom2014 solar Zone 3**, an Australian PV series independent of the building
in Part a. Predict normalised PV power in p.u. one hour ahead using the previous
PV observation and weather forecasts for the target hour. This notebook runs independently.

Check when predictors are available, fit a linear model, apply prediction
bounds, and compare persistence, regression, a decision tree and a random forest. Reserve calibration data for the next session.
Tutorial 4.2 uses linear regression to examine forecast uncertainty;
its model choice is specified before test evaluation.

Run from the repository root. Allow **35–45 minutes**: setup/availability 10,
features/models/corrections 20, scoring/interpretation 10, calibration/test discussion 5 minutes.
Complete six short TODO cells, including fitting all three models. Probabilistic forecasts are covered on Day 2.
Source: [Zone 3 extract](data/tutorial_4_2/README.md) and [Day 1 preparation](data/week4_day1/README.md).
""", "overview", tags=[])

cell("markdown", r"""
## P0. Load the data and define chronological splits (slide 31)

The data include forecast irradiance (W/m²), forecast air temperature (°C),
forecast cloud cover (fraction) and observed PV power (p.u.).
Retain the weather forecast's issue time and valid time. Irradiance has already been
calculated from differences in accumulated radiation, divided by 3600 seconds.
Do not difference this column again. All timestamps are in **UTC**.
The exact location is undisclosed, so 12:00 UTC should not be described as local midday.

One source value slightly exceeds 1 p.u. Preserve the original values in
`solar_power_pu_raw`, and create a separate target constrained to [0,1] before calculating
lags. Report the values changed by this bound. This bound is a modelling choice for the exercise:
normalised output can exceed nominal capacity in the original data.

Training: 1 April 2012 01:00–31 March 2013 23:00; calibration: April–June 2013;
test: July–September 2013, all UTC. Do not use calibration or test targets to fit the model.
""", "p0", tags=[])

cell("code", r"""
from pathlib import Path
import hashlib
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

plt.rcParams.update({"figure.dpi": 100, "font.size": 12,
                     "axes.grid": True, "grid.alpha": 0.2, "svg.fonttype": "none"})
ROOT = Path.cwd()
if not (ROOT / "data").exists():
    raise FileNotFoundError("Launch Jupyter from the repository root.")
FIGURE_DIR = ROOT / "figures" / "week4_day1"
EXPORT_ARTEFACTS = True  # Set False to display plots without exporting files.
MODEL_COLOURS = {"Persistence": "tab:orange", "Linear regression": "tab:blue",
                 "Decision tree": "tab:red", "Random forest": "tab:green"}
SLIDE_NUMBERS = {12: 31, 13: 32, 14: 34, 15: 34, 16: 35}
SPEC_SLIDE_NUMBERS = SLIDE_NUMBERS.copy()

NOTEBOOK_NAME = "tutorial_4_1b_pv_point_forecasting_solution.ipynb"
SOURCE_LABEL = "GEFCom2014 solar Zone 3; UTC"
SOURCE_SHA256 = hashlib.sha256((ROOT / "data/tutorial_4_2/gefcom2014_solar_zone3.csv").read_bytes()).hexdigest()
""", "setup", tags=["supplied"])

cell("markdown", r"""
### Supplied plotting helpers

The next two functions keep figure styles and exports consistent. Data preparation,
model fitting and error calculations are shown in the subsequent cells.
""", "plotting_helpers_intro", tags=[])

cell("code", r"""
# Supplied file/plot helper. All feature and model calculations stay in later cells.
def export_figure(fig, number, slug, checkpoint, period, units, description, n):
    fig.text(0.01, 0.01, textwrap.fill(
        f"{checkpoint} | {SOURCE_LABEL} | {period} | n={n} | {units}", 118), fontsize=9)
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    if EXPORT_ARTEFACTS:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"w4d1_F{number:02d}_{slug}.png"
        fig.savefig(FIGURE_DIR / filename, dpi=220, bbox_inches="tight", facecolor="white")
        fig.savefig(FIGURE_DIR / filename.replace(".png", ".svg"), bbox_inches="tight", facecolor="white")
        manifest_path = ROOT / "teaching" / "week4_day1_figure_manifest.csv"
        record = pd.DataFrame([{
            "figure_id": f"F{number:02d}", "filename": f"figures/week4_day1/{filename}",
            "slide_number": SLIDE_NUMBERS[number], "spec_slide_number": SPEC_SLIDE_NUMBERS[number],
            "notebook": NOTEBOOK_NAME, "checkpoint": checkpoint, "data_period": period,
            "units": units, "description": description, "status": "generated",
            "optional": number == 17, "sample_count": n, "source_sha256": SOURCE_SHA256,
        }])
        old = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
        if not old.empty:
            old = old.loc[old.figure_id != f"F{number:02d}"]
        pd.concat([old, record], ignore_index=True).sort_values("figure_id").to_csv(manifest_path, index=False)
    plt.show()
    plt.close(fig)

""", "figure_export", tags=["supplied"])

cell("code", r"""
def draw_table(ax, table, widths=None, fontsize=11):
    ax.axis("off")
    table = table.copy()
    for column in table.select_dtypes(include="floating"):
        table[column] = table[column].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "undefined")
    artist = ax.table(cellText=table.to_numpy(), colLabels=table.columns,
                      loc="center", cellLoc="center", colWidths=widths)
    artist.auto_set_font_size(False)
    artist.set_fontsize(fontsize)
    artist.scale(1, 1.8)
    for (row, column), cell in artist.get_celld().items():
        cell.set_edgecolor("0.85")
        if row == 0:
            cell.set_facecolor("#24476a")
            cell.set_text_props(color="white", weight="bold")
    return artist
""", "table_plotting", tags=["supplied"])

cell("code", r"""
PV_PATH = ROOT / "data" / "tutorial_4_2" / "gefcom2014_solar_zone3.csv"
pv_source = pd.read_csv(PV_PATH, index_col="timestamp_utc")
pv_source.index = pd.to_datetime(pv_source.index, utc=True)
pv_source = pv_source.sort_index()
pv_source["forecast_issue_time_utc"] = pd.to_datetime(pv_source["forecast_issue_time_utc"], utc=True)
expected_pv_index = pd.date_range(pv_source.index.min(), pv_source.index.max(), freq="h")
if not pv_source.index.is_unique or not pv_source.index.equals(expected_pv_index):
    raise ValueError("PV lag construction requires a complete, unique hourly index.")
WEATHER_COLUMNS = ["forecast_surface_solar_radiation_w_m2", "forecast_air_temperature_c",
                   "forecast_total_cloud_cover_fraction"]
# Explicit mapping: these descriptive source names are already in the repository CSV.
pv = pv_source[["solar_power_pu", *WEATHER_COLUMNS,
                "forecast_issue_time_utc"]].copy()
pv["solar_power_pu_raw"] = pv["solar_power_pu"].copy()
pv["solar_power_pu"] = pv["solar_power_pu_raw"].clip(lower=0, upper=1)
changed_targets = pv.loc[pv.solar_power_pu != pv.solar_power_pu_raw,
                         ["solar_power_pu_raw", "solar_power_pu"]]
pv["origin_time_utc"] = pv.index - pd.Timedelta(hours=1)
pv["solar_available"] = pv[WEATHER_COLUMNS[0]] > 1.0
if (pv["forecast_issue_time_utc"] > pv["origin_time_utc"]).any():
    raise ValueError("A weather forecast was issued after its PV forecast origin.")
""", "read_and_contract", tags=["supplied"])

cell("markdown", r"""
### Reserve training, calibration and test periods

Define the three periods by target timestamp. Train the models on the first period
only. Reserve calibration data for the uncertainty analysis in Tutorial 4.2.
""", "period_definitions-intro", tags=[])

cell("code", r"""
PV_TRAIN_START, PV_TRAIN_END = "2012-04-01 01:00+00:00", "2013-03-31 23:00+00:00"
PV_CAL_START, PV_CAL_END = "2013-04-01 00:00+00:00", "2013-06-30 23:00+00:00"
PV_TEST_START, PV_TEST_END = "2013-07-01 00:00+00:00", "2013-09-30 23:00+00:00"
train_index = pv.loc[pd.Timestamp(PV_TRAIN_START):pd.Timestamp(PV_TRAIN_END)].index
calibration_index = pv.loc[pd.Timestamp(PV_CAL_START):pd.Timestamp(PV_CAL_END)].index
test_index = pv.loc[pd.Timestamp(PV_TEST_START):pd.Timestamp(PV_TEST_END)].index

print(f"Source: {len(pv_source):,} rows from {pv_source.index.min()} to {pv_source.index.max()}")
print("Missing required readings:", pv[["solar_power_pu", *WEATHER_COLUMNS]].isna().sum().to_dict())
print("Altered teaching targets:", len(changed_targets))
display(changed_targets)
print("Training/calibration/test before lag removal:", len(train_index), len(calibration_index), len(test_index))
display(pv.loc[train_index[:3], ["solar_power_pu", *WEATHER_COLUMNS]])
""", "period_definitions", tags=["supplied"])

cell("markdown", r"""
**Predict:** will the PV/irradiance peaks occur around 12:00 UTC? Examine the three training
days below. Then describe the relationship and one limitation of the weather forecasts.
Irradiance here is a weather forecast, not a measurement of actual radiation.
""", "profile_question", tags=[])

cell("code", r"""
profile_index = pv.loc["2012-05-10":"2012-05-12 23:00"].index
fig, axes = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
axes[0].plot(profile_index, pv.loc[profile_index, "solar_power_pu"], color="black", label="Observed PV (teaching target)")
axes[0].set(title="Three training days; Australian daylight need not match UTC daytime", ylabel="PV power (p.u.)")
axes[0].legend(fontsize=10)
axes[1].plot(profile_index, pv.loc[profile_index, WEATHER_COLUMNS[0]], color="tab:blue", label="Forecast irradiance")
axes[1].set(ylabel="Forecast irradiance (W/m²)", xlabel="Target time (UTC)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=pv.index.tz))
axes[1].legend(fontsize=10)
export_figure(fig, 12, "pv_profile", "P0", "10–12 May 2012 UTC",
              "PV p.u.; forecast W/m²", "Separate time-aligned PV and forecast-radiation panels", len(profile_index))
""", "profile_plot", tags=["supplied"])

cell("markdown", r"""
## P1. Construct the previous-hour predictor (slide 32)

For a target at $t$, issue the forecast at $t-1$ hour. At that time, we assume the
previous PV observation is available. A weather forecast for $t$ can be used only
if it was issued by $t-1$.

We assume weather forecasts are available immediately after issue because the data
do not record publication delays. In practice, check both publication
and measurement delays.

**Task 1:** create `pv_lag_1` from the [0,1] target, using the complete hourly index.
An earlier calibration/test observation can be used at a later forecast origin.
We issue a new one-hour-ahead forecast each hour; we do not forecast a whole day
from one origin.
""", "p1", tags=[])

cell("code", r"""
pv_features = pv[WEATHER_COLUMNS].copy()
pv_features.insert(0, "pv_lag_1", pv["solar_power_pu"].shift(1))
""", "pv_features", tags=["exercise"], student=r"""
pv_features = pv[WEATHER_COLUMNS].copy()
# TODO: add pv_lag_1 by shifting the bounded solar_power_pu target by one hour.
# Keep the column first so the input order is [pv_lag_1, *WEATHER_COLUMNS].
raise NotImplementedError("Construct the available one-hour PV lag.")
""")

cell("code", r"""
PV_FEATURE_COLUMNS = ["pv_lag_1", *WEATHER_COLUMNS]
X_pv_train = pv_features.loc[train_index].dropna()
X_pv_calibration = pv_features.loc[calibration_index]
X_pv_test = pv_features.loc[test_index]
y_pv_train = pv.loc[X_pv_train.index, "solar_power_pu"]
y_pv_calibration = pv.loc[calibration_index, "solar_power_pu"]
y_pv_test = pv.loc[test_index, "solar_power_pu"]

print("Retained training/calibration/test:", len(y_pv_train), len(y_pv_calibration), len(y_pv_test))
print("One excluded initial training row; no excluded calibration/test targets.")
""", "pv_split", tags=["supplied"])

cell("code", r"""
example_targets = pd.DatetimeIndex(["2013-07-01 00:00+00:00", "2013-07-01 01:00+00:00", "2013-07-01 12:00+00:00"])
example = pv.loc[example_targets]
mapping = pd.DataFrame({
    "Target / valid UTC": example_targets.strftime("%d %b\n%H:%M"),
    "Origin UTC": example.origin_time_utc.dt.strftime("%d %b\n%H:%M").to_numpy(),
    "PV lag source UTC": (example_targets - pd.Timedelta(hours=1)).strftime("%d %b\n%H:%M"),
    "Observed lag (p.u.)": pv_features.loc[example_targets, "pv_lag_1"].round(3).to_numpy(),
    "Weather issued UTC": example.forecast_issue_time_utc.dt.strftime("%d %b\n%H:%M").to_numpy(),
})
display(mapping)
fig, axes = plt.subplots(2, 1, figsize=(13, 5.8), gridspec_kw={"height_ratios": [1, 1.2]})
first = example.iloc[0]
times = [first.forecast_issue_time_utc, first.origin_time_utc, example_targets[0]]
axes[0].plot(times, [0, 0, 0], color="0.6", marker="o", markersize=8)
for time, label, offset in zip(times, ["Weather issue\n30 Jun 00:00", "Origin + PV lag source\n30 Jun 23:00", "Target / valid time\n1 Jul 00:00"], [(0, 15), (-80, -45), (15, 15)]):
    axes[0].annotate(label, (time, 0), xytext=offset, textcoords="offset points", fontsize=10)
axes[0].set(yticks=[], ylim=(-0.6, 0.6), title="Issue ≤ origin < target; the final step is one hour")
axes[0].set_xlim(pd.Timestamp("2013-06-29 22:00", tz="UTC"), pd.Timestamp("2013-07-01 04:00", tz="UTC"))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=pv.index.tz))
draw_table(axes[1], mapping, fontsize=11)
for cell in axes[1].tables[0].get_celld().values():
    cell.set_height(0.2)
export_figure(fig, 13, "pv_origin_weather", "P1", "1 July 2013: 00:00, 01:00 and 12:00 UTC",
              "UTC timestamps; PV p.u.", "Three actual source/issue/origin/valid-time examples", 3)
""", "origin_weather_plot", tags=["supplied"])

cell("markdown", r"""
## P2. Fit the models and generate forecasts (slide 33)

### Worked example: persistence

Persistence repeats the previous PV observation: $\hat y_{t\mid t-1}=y_{t-1}$.

Constrain predictions to [0,1], then set them to zero when forecast irradiance is
at most 1 W/m². Apply these steps to every forecasting method.
`solar_available` indicates whether forecast irradiance exceeds this threshold.
It estimates when solar production is possible, but can be wrong: actual PV may be
positive when forecast irradiance is low. The rule uses only forecast information,
not the target's observed PV. Retain both raw and corrected predictions.
""", "p2", tags=[])

cell("code", r"""
persistence_calibration_raw = X_pv_calibration["pv_lag_1"].copy()
persistence_test_raw = X_pv_test["pv_lag_1"].copy()
persistence_calibration = persistence_calibration_raw.clip(lower=0, upper=1)
persistence_test = persistence_test_raw.clip(lower=0, upper=1)
persistence_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
persistence_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0
""", "persistence", tags=["supplied"])

cell("markdown", r"""
### Task 2: linear regression with weather forecasts

Predict $\hat y_t=\beta_0+\beta_1 y_{t-1}+\beta_R R_t^{\mathrm{fc}}+\beta_T T_t^{\mathrm{fc}}+\beta_C C_t^{\mathrm{fc}}$.

Here, $y_{t-1}$ is the previous PV observation, and $R_t^{\mathrm{fc}}$,
$T_t^{\mathrm{fc}}$ and $C_t^{\mathrm{fc}}$ are irradiance, temperature and cloud-cover
forecasts for the target hour. Each coefficient weights one predictor;
$\beta_0$ is the intercept. The output is in p.u.

Fit the model once using training data, then predict calibration and test targets
without refitting.

**Predict:** will weather forecasts help predict changes in PV output compared with persistence?
""", "linear_description", tags=[])

cell("code", r"""
pv_linear_model = LinearRegression()
pv_linear_model.fit(X_pv_train[PV_FEATURE_COLUMNS], y_pv_train)
fitted_coefficients = pv_linear_model.coef_.copy()
linear_calibration_raw = pd.Series(pv_linear_model.predict(X_pv_calibration[PV_FEATURE_COLUMNS]), index=calibration_index)
linear_test_raw = pd.Series(pv_linear_model.predict(X_pv_test[PV_FEATURE_COLUMNS]), index=test_index)
np.testing.assert_array_equal(fitted_coefficients, pv_linear_model.coef_)
""", "linear_fit", tags=["exercise"], student=r"""
pv_linear_model = LinearRegression()
# TODO: fit on X_pv_train[PV_FEATURE_COLUMNS]/y_pv_train.
# Predict calibration and test; create linear_calibration_raw and linear_test_raw
# as Series indexed by calibration_index and test_index. Do not refit.
raise NotImplementedError("Fit the linear PV model and generate predictions before applying bounds.")
""")

cell("markdown", r"""
### Task 3: apply PV prediction bounds

Constrain calibration and test forecasts to [0,1]: replace negative predictions with
zero and predictions above 1 with 1. Then set forecasts to zero where
`solar_available=False`, as for persistence.

Apply these rules to predictions only. Keep observed targets unchanged, including
positive output at times when forecast irradiance is low.
""", "projection_description", tags=[])

cell("code", r"""
linear_calibration = linear_calibration_raw.clip(lower=0, upper=1)
linear_test = linear_test_raw.clip(lower=0, upper=1)
linear_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
linear_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0
""", "linear_correction", tags=["exercise"], student=r"""
# TODO: create linear_calibration and linear_test from the raw forecasts.
# Constrain forecasts to [0,1] and set them to zero when forecast irradiance is low.
raise NotImplementedError("Correct the linear forecasts consistently with persistence.")
""")

cell("markdown", r"""
### Task 4: fit a PV decision tree

Use the same four predictors as linear regression: previous-hour PV, target-hour
forecast irradiance, temperature and cloud cover. A decision tree learns threshold
rules and predicts the mean training output within each final region.
It can represent nonlinear relationships between weather and PV output.

Fit once on training data, without scaling. Fix maximum depth at 6 and at least four
cases per leaf before test evaluation. Apply the same [0,1] and low-irradiance
corrections used for persistence and regression.

**Predict:** when might a tree help with PV changes that a linear model misses?

Follow Task 2: fit on the training inputs/targets and predict both reserved periods.
Store the model in `pv_tree` and the indexed raw forecasts in
`tree_calibration_raw` and `tree_test_raw`. The correction lines are supplied.
""", "pv_tree_description", tags=[])

cell("code", r"""
pv_tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42)
pv_tree.fit(X_pv_train[PV_FEATURE_COLUMNS], y_pv_train)
tree_calibration_raw = pd.Series(pv_tree.predict(X_pv_calibration[PV_FEATURE_COLUMNS]), index=calibration_index)
tree_test_raw = pd.Series(pv_tree.predict(X_pv_test[PV_FEATURE_COLUMNS]), index=test_index)

tree_calibration = tree_calibration_raw.clip(lower=0, upper=1)
tree_test = tree_test_raw.clip(lower=0, upper=1)
tree_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
tree_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0
""", "pv_tree_fit", tags=["exercise"], student=r"""
pv_tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=4, random_state=42)
# TODO: fit pv_tree on X_pv_train[PV_FEATURE_COLUMNS] and y_pv_train.
# Create tree_calibration_raw and tree_test_raw using predict on the matching inputs.
# Store them as Series indexed by calibration_index and test_index, respectively.
# Follow Task 2's structure; do not fit on calibration or test.
raise NotImplementedError("Fit the PV decision tree and generate aligned calibration/test predictions.")

# Supplied corrections: the same rules as Task 3.
tree_calibration = tree_calibration_raw.clip(lower=0, upper=1)
tree_test = tree_test_raw.clip(lower=0, upper=1)
tree_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
tree_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0
""")

cell("markdown", r"""
### Task 5: fit a PV random forest

A random forest averages many trees fitted to resampled training rows.
At each split it considers a random subset of predictors. Use exactly the same
four predictors and training targets as the decision tree, without scaling.

The settings below are fixed before scoring. Retain raw predictions, then apply
the same corrections as every other method. Neither calibration nor test data are used to fit the model.

**Predict:** will averaging trees improve on a single tree in this test period?
Use the common error table rather than assuming a ranking.

Tutorial 4.2 uses linear regression, regardless of the ranking in this comparison.
The calendar-encoding example is in Part a. This PV example uses no calendar variables.

Complete the fit/predict lines using the same feature order as the tree.
Store the model in `pv_forest` and the indexed raw forecasts in
`forest_calibration_raw` and `forest_test_raw`. Use the same fitted model for both periods.
""", "pv_forest_description", tags=[])

cell("code", r"""
pv_forest = RandomForestRegressor(n_estimators=150, min_samples_leaf=4, max_features=0.8,
                                  random_state=42, n_jobs=1)
pv_forest.fit(X_pv_train[PV_FEATURE_COLUMNS], y_pv_train)
forest_calibration_raw = pd.Series(pv_forest.predict(X_pv_calibration[PV_FEATURE_COLUMNS]), index=calibration_index)
forest_test_raw = pd.Series(pv_forest.predict(X_pv_test[PV_FEATURE_COLUMNS]), index=test_index)

forest_calibration = forest_calibration_raw.clip(lower=0, upper=1)
forest_test = forest_test_raw.clip(lower=0, upper=1)
forest_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
forest_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0
""", "pv_forest_fit", tags=["exercise"], student=r"""
pv_forest = RandomForestRegressor(n_estimators=150, min_samples_leaf=4, max_features=0.8,
                                  random_state=42, n_jobs=1)
# TODO: fit pv_forest on the SAME training predictors/targets as the PV decision tree.
# Create forest_calibration_raw and forest_test_raw as indexed Series.
# Reuse the fitted forest for both periods; do not refit.
raise NotImplementedError("Fit the PV random forest and generate aligned calibration/test predictions.")

# Supplied corrections: keep raw and corrected predictions separate.
forest_calibration = forest_calibration_raw.clip(lower=0, upper=1)
forest_test = forest_test_raw.clip(lower=0, upper=1)
forest_calibration.loc[~pv.loc[calibration_index, "solar_available"]] = 0.0
forest_test.loc[~pv.loc[test_index, "solar_available"]] = 0.0
""")

cell("code", r"""
pv_forecasts = pd.DataFrame({"Persistence": persistence_test, "Linear regression": linear_test,
                              "Decision tree": tree_test, "Random forest": forest_test}, index=test_index)
raw_forecasts = pd.DataFrame({"Persistence": persistence_test_raw, "Linear regression": linear_test_raw,
                             "Decision tree": tree_test_raw, "Random forest": forest_test_raw})

correction_rows = []
for name in pv_forecasts:
    raw = raw_forecasts[name]
    correction_rows.append({"Method": name,
                            "Raw outside [0,1]": int(((raw < 0) | (raw > 1)).sum()),
                            "Changed test forecasts": int(pv_forecasts[name].ne(raw).sum())})
display(pd.DataFrame(correction_rows))
""", "combine_predictions", tags=["supplied"])

cell("markdown", r"""
## P3. Ramps, missed peaks and forecast errors (slide 34)

**Task 6:** use the supplied functions to calculate full-test MAE and RMSE for all four methods.
Pass actuals first and forecasts second. Both metrics are in p.u.
Do not calculate PV MAPE because actual output is often zero.

**Predict:** could many zero-output hours make full-day errors appear small?

The supplied second table evaluates the same models when forecast irradiance exceeds
1 W/m². Report both subsets with their counts. This subset uses forecast information,
not observed PV, and is not a perfectly known daylight mask.

The weekly plot is enlarged around the largest observed one-hour change in that
displayed week. Selection is solely for diagnosis, not for model fitting or tuning.
Compare the timing and magnitude of the ramp. What does a point prediction leave unspecified?
""", "p3", tags=[])

cell("code", r"""
def metric_arrays(actual, forecast):
    '''Check paired Series before converting them to numerical arrays.'''
    if not isinstance(actual, pd.Series) or not isinstance(forecast, pd.Series):
        raise TypeError("Pass actual and forecast as pandas Series.")
    if len(actual) == 0 or len(actual) != len(forecast) or not actual.index.equals(forecast.index):
        raise ValueError("Actual and forecast must have equal, non-empty aligned timestamps.")
    if not actual.index.is_unique:
        raise ValueError("Metric timestamps must be unique.")
    a, f = actual.to_numpy(dtype=float), forecast.to_numpy(dtype=float)
    if not np.isfinite(a).all() or not np.isfinite(f).all():
        raise ValueError("Resolve/count non-finite inputs upstream; metrics never drop pairs.")
    return a, f

def mae(actual, forecast):
    a, f = metric_arrays(actual, forecast)
    return float(np.mean(np.abs(a - f)))

def rmse(actual, forecast):
    a, f = metric_arrays(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2)))
""", "metrics", tags=["supplied"])

cell("code", r"""
pv_rows = []
for name in pv_forecasts:
    prediction = pv_forecasts[name]
    pv_rows.append({"Model": name, "MAE (p.u.)": mae(y_pv_test, prediction),
                    "RMSE (p.u.)": rmse(y_pv_test, prediction), "n": len(y_pv_test)})
pv_comparison = pd.DataFrame(pv_rows).set_index("Model")
display(pv_comparison.round(3))
""", "comparison", tags=["exercise"], student=r"""
# TODO: create pv_comparison with models as rows and MAE (p.u.), RMSE (p.u.), n.
# Use the complete common y_pv_test and each corrected pv_forecasts column.
raise NotImplementedError("Compare all four models on the same test targets.")
""")

cell("code", r"""
solar_subset = pv.loc[test_index, "solar_available"]
pv_subset_rows = []
for name in pv_forecasts:
    # Reuse the all-hour student calculation. Compute the disclosed secondary subset.
    pv_subset_rows.append({"Subset": "All hours", "Model": name,
                          "MAE (p.u.)": pv_comparison.loc[name, "MAE (p.u.)"],
                          "RMSE (p.u.)": pv_comparison.loc[name, "RMSE (p.u.)"], "n": len(y_pv_test)})
    pv_subset_rows.append({"Subset": "Forecast irradiance >1 W/m²", "Model": name,
                          "MAE (p.u.)": mae(y_pv_test.loc[solar_subset], pv_forecasts.loc[solar_subset, name]),
                          "RMSE (p.u.)": rmse(y_pv_test.loc[solar_subset], pv_forecasts.loc[solar_subset, name]),
                          "n": int(solar_subset.sum())})
pv_subset_comparison = pd.DataFrame(pv_subset_rows)
display(pv_subset_comparison.round(3))
""", "subset_scores", tags=["supplied"])

cell("code", r"""
pv_plot_index = y_pv_test.loc["2013-07-15":"2013-07-21 23:00"].index
# Select the largest observed one-hour change WITHIN the displayed week.
week_changes = y_pv_test.loc[pv_plot_index].diff().abs()
ramp_time = week_changes.idxmax()
ramp_index = y_pv_test.loc[ramp_time - pd.Timedelta(hours=6):ramp_time + pd.Timedelta(hours=6)].index
fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), gridspec_kw={"height_ratios": [1, 1.1]})
for ax, index in zip(axes, [pv_plot_index, ramp_index]):
    ax.plot(index, y_pv_test.loc[index], color="black", linewidth=2, label="Actual PV")
    for name in pv_forecasts:
        ax.plot(index, pv_forecasts.loc[index, name], color=MODEL_COLOURS[name],
                linewidth=1.3, marker="o" if ax == axes[1] else None, markersize=4, label=name)
    ax.set(ylabel="PV power (p.u.)", xlabel="Target time (UTC)", ylim=(-0.03, 1.03))
    ax.legend(ncol=3, fontsize=9)
axes[0].set_title("One-hour-ahead forecasts: weekly overview")
axes[0].axvspan(ramp_index.min(), ramp_index.max(), color="0.5", alpha=0.15, label="Enlarged below")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=pv.index.tz))
axes[1].set_title(f"Largest observed hourly change in display week: {ramp_time:%d %b %Y %H:%M} UTC")
axes[1].axvline(ramp_time, color="0.5", linestyle="--")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=pv.index.tz))
print("Ramp change:", y_pv_test.diff().loc[ramp_time], "p.u.; target:", ramp_time)
export_figure(fig, 14, "pv_forecasts", "P3", "15–21 July 2013 UTC; largest-ramp close-up",
              "PV p.u.", "Weekly overview and observed ramp selected transparently for diagnosis", len(pv_plot_index))
""", "forecast_plot", tags=["supplied"])

cell("code", r"""
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_title("Point forecast scores: all test hours and forecast-solar-available hours", pad=18)
draw_table(ax, pv_subset_comparison.round(3), widths=[0.33, 0.27, 0.16, 0.16, 0.08], fontsize=11)
export_figure(fig, 15, "pv_scores", "P3", "1 July–30 September 2013 UTC",
              "MAE/RMSE p.u.", "All hours versus forecast irradiance >1 W/m²; common targets and counts", len(y_pv_test))
""", "scores_plot", tags=["supplied"])

cell("code", r"""
best = pv_comparison["RMSE (p.u.)"].idxmin()
print(f"{best} has the lowest all-hour RMSE ({pv_comparison.loc[best, 'RMSE (p.u.)']:.3f} p.u.).")
print(f"The displayed ramp occurs at {ramp_time:%d %b %H:%M} UTC. Compare the enlarged observed and predicted changes rather than judging only the weekly overview.")
print(f"Forecast-solar-available hours: n={int(solar_subset.sum())}, versus {len(y_pv_test)} all-hour targets. The subset removes many easy low-output hours, so its scores answer a different question.")
print("The solar-availability proxy can be wrong. Point forecasts specify one value, not a range of plausible outcomes.")
print("Tutorial 4.2 uses errors from corrected linear calibration forecasts to construct prediction intervals.")
print("The tree and forest share the same PV/weather features and correction rules. Their nonlinear predictions can differ from regression, without changing the model choice specified for Tutorial 4.2.")
""", "worked_discussion", tags=["worked-discussion"])

cell("markdown", r"""
## P4. Calibration and forecast uncertainty (slide 35)

The diagram summarises the three chronological periods. The linear model is trained
once, then used to predict the calibration and test targets.

Tutorial 4.2 constructs prediction intervals from calibration errors:
**actual minus the corrected forecast**. It rebuilds this model directly from source
data, so no saved model, prediction bundle or previous notebook state is required.
Do not calculate intervals or coverage in this point-forecasting tutorial.
""", "p4", tags=[])

cell("code", r"""
fig, ax = plt.subplots(figsize=(12, 3.7))
periods = [("Training", PV_TRAIN_START, PV_TRAIN_END, len(y_pv_train), "tab:blue"),
           ("Calibration", PV_CAL_START, PV_CAL_END, len(y_pv_calibration), "tab:green"),
           ("Test", PV_TEST_START, PV_TEST_END, len(y_pv_test), "tab:orange")]
for row, (label, start, end, count, colour) in enumerate(periods):
    left = mdates.date2num(pd.Timestamp(start))
    right = mdates.date2num(pd.Timestamp(end) + pd.Timedelta(hours=1))
    ax.barh(row, right - left, left=left, height=0.55, color=colour)
ax.set(yticks=range(3), yticklabels=[f"{label}: {count:,} targets" for label, _, _, count, _ in periods],
       title="Fit linear regression once; reuse unchanged for calibration and test", xlabel="Target date (UTC)")
ax.text(0.01, 0.93, "Train: Apr 2012–Mar 2013; calibration: Apr–Jun 2013; test: Jul–Sep 2013",
        transform=ax.transAxes, fontsize=10)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=pv.index.tz))
export_figure(fig, 16, "pv_split_handover", "P4", "1 April 2012–30 September 2013 UTC",
              "timestamps; target counts", "Fixed linear model; reserved calibration; no intervals yet", len(y_pv_train) + len(y_pv_calibration) + len(y_pv_test))
""", "handover_plot", tags=["supplied"])

cell("code", r"""
# Optional instructor exports; Tutorial 4.2 does not load these files.
if EXPORT_ARTEFACTS:
    result_dir = ROOT / "teaching/week4_day1_outputs"
    result_dir.mkdir(parents=True, exist_ok=True)
    pv_forecasts.to_csv(result_dir / "pv_test_forecasts.csv")
    pv_comparison.to_csv(result_dir / "pv_scores.csv")
    pv_subset_comparison.to_csv(result_dir / "pv_subset_scores.csv", index=False)
""", "export_results", tags=["supplied"])

cell("markdown", r"""
**Exit question:** why can PV lag 1 be used at every hourly origin, while observed load
lag 1 is unavailable for most targets at a common day-ahead origin?
""", "exit_question", tags=[])

pairs["tutorial_4_1b_pv_point_forecasting"] = (student_cells, solution_cells)

if __name__ == "__main__":
    student_only = "--student-only" in sys.argv
    for name, (student_sources, solution_sources) in pairs.items():
        versions = [("", student_sources)] if student_only else [("", student_sources), ("_solution", solution_sources)]
        for suffix, cells in versions:
            notebook = nbf.v4.new_notebook(cells=cells, metadata={
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}})
            nbf.validate(notebook)
            nbf.write(notebook, ROOT / (name + suffix + ".ipynb"))
            print(name + suffix, len(cells), "cells")
