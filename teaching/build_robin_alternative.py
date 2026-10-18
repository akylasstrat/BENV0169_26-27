"""Prepare compact BDG2 extracts and create the alternative UCL notebook pair.

Instructor use only. Run from this repository root with the local BDG2 checkout
as the sole argument. Source data and the existing notebooks are not modified.
The student notebooks do not depend on this authoring utility.
"""
import argparse
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("bdg2_repository", type=Path)
parser.add_argument("--data-only", action="store_true", help="Rebuild extracts without replacing edited notebooks.")
parser.add_argument("--legacy-notebooks", action="store_true",
                    help="Explicitly rebuild the old notebook template, replacing lecture revisions. Not the current teaching route.")
args = parser.parse_args()
root = Path.cwd()
source_root = args.bdg2_repository.resolve()
data_dir = root / "data/week4_day1/robin"
data_dir.mkdir(parents=True, exist_ok=True)
building = "Robin_education_Margarito"
period_start, period_end = "2017-01-08 00:00", "2017-09-30 23:00"
source_paths = {
    "metadata": source_root / "data/metadata/metadata.csv",
    "electricity": source_root / "data/meters/cleaned/electricity_cleaned.csv",
    "hotwater": source_root / "data/meters/cleaned/hotwater_cleaned.csv",
    "weather": source_root / "data/weather/weather.csv",
}
metadata = pd.read_csv(source_paths["metadata"])
robin = metadata.loc[metadata.site_id.eq("Robin")].copy()
categories = ["Lodging/residential", "Education", "Office"]
profile_buildings = robin.loc[robin.primaryspaceusage.isin(categories)].copy()
electricity = pd.read_csv(source_paths["electricity"], index_col="timestamp",
                          parse_dates=True, usecols=["timestamp", *robin.building_id])
heat_buildings = robin.loc[robin.hotwater.eq("Yes"), "building_id"].tolist()
hotwater = pd.read_csv(source_paths["hotwater"], index_col="timestamp",
                       parse_dates=True, usecols=["timestamp", *heat_buildings])
weather = pd.read_csv(source_paths["weather"])
weather = weather.loc[weather.site_id.eq("Robin")].copy()
weather.index = pd.to_datetime(weather.timestamp)
assert electricity.index.is_unique and hotwater.index.is_unique and weather.index.is_unique
grid = pd.date_range("2017-01-01", "2017-12-31 23:00", freq="h", name="timestamp")
extract = pd.DataFrame(index=grid)
extract["electricity_kwh"] = electricity[building].reindex(grid)
extract["air_temperature_c"] = weather.airTemperature.reindex(grid)
extract["hotwater_kwh"] = hotwater[building].reindex(grid)
assert extract.electricity_kwh.notna().all()
selected_runs = extract.electricity_kwh.groupby(extract.electricity_kwh.ne(extract.electricity_kwh.shift()).cumsum()).size()
assert selected_runs.max() == 2
extract.to_csv(data_dir / "robin_education_margarito_2017.csv")

# Compare source quality before model fitting. No forecast scores are used here.
quality_rows = []
for candidate in heat_buildings:
    for year in [2016, 2017]:
        history = electricity.loc[f"{year}-01-01":f"{year}-10-28 23:00", candidate]
        training = history.loc[f"{year}-01-08":f"{year}-09-30 23:00"]
        test = history.loc[f"{year}-10-01":f"{year}-10-28 23:00"]
        runs = history.groupby(history.ne(history.shift()).cumsum()).size()
        quality_rows.append({"building_id": candidate, "year": year,
                             "training_missing": int(training.isna().sum()),
                             "test_missing": int(test.isna().sum()), "test_nonpositive": int(test.le(0).sum()),
                             "flat_hours_24plus": int(runs.loc[runs >= 24].sum()),
                             "longest_flat_hours": int(runs.max())})
pd.DataFrame(quality_rows).to_csv(data_dir / "building_quality_comparison.csv", index=False)
metadata_columns = ["building_id", "site_id", "primaryspaceusage", "sub_primaryspaceusage",
                    "sqm", "timezone", "electricity", "hotwater", "gas"]
robin[metadata_columns].to_csv(data_dir / "robin_building_metadata.csv", index=False)

# Training-only preparation for the usage-type display. No gap filling.
profile_rows = []
profile_quality = []
for row in profile_buildings.itertuples():
    original = electricity.loc[period_start:period_end, row.building_id]
    run_lengths = original.groupby(original.ne(original.shift()).cumsum()).transform("size")
    flat_flag = original.notna() & run_lengths.ge(24)
    readings = original.mask(flat_flag)
    profile_quality.append({"building_id": row.building_id, "usage_type": row.primaryspaceusage,
                            "source_missing": int(original.isna().sum()), "flagged_flat_hours": int(flat_flag.sum()),
                            "retained_hours": int(readings.notna().sum())})
    scale = readings.max()
    if not np.isfinite(scale) or scale <= 0:
        continue  # No usable varying readings remain for this building.
    frame = pd.DataFrame({"electricity_kwh": readings})
    frame["hour"] = frame.index.hour
    frame["day_type"] = np.where(frame.index.dayofweek < 5, "Weekday", "Weekend")
    grouped = frame.groupby(["day_type", "hour"]).electricity_kwh.agg(["mean", "count"])
    for (day_type, hour), values in grouped.iterrows():
        profile_rows.append({"building_id": row.building_id, "usage_type": row.primaryspaceusage,
                             "day_type": day_type, "hour": hour,
                             "mean_electricity_kwh": values["mean"], "n_observed": int(values["count"]),
                             "training_max_kwh": scale, "normalised_mean": values["mean"] / scale})
profiles = pd.DataFrame(profile_rows)
assert len(profiles) == profiles.building_id.nunique() * 2 * 24
assert profiles.normalised_mean.notna().all() and profiles.n_observed.gt(0).all()
profiles.to_csv(data_dir / "robin_usage_profiles.csv", index=False)
profile_quality = pd.DataFrame(profile_quality)
profile_quality.to_csv(data_dir / "robin_profile_quality.csv", index=False)
# The revised lecture's fixed group is selected by usage and training quality.
group_ids = [building]
for category in ["Lodging/residential", "Office"]:
    candidates = sorted(robin.loc[robin.primaryspaceusage.eq(category), "building_id"])
    for candidate in candidates:
        readings = electricity.loc[period_start:period_end, candidate]
        longest_run = readings.groupby(readings.ne(readings.shift()).cumsum()).size().max()
        if readings.notna().all() and longest_run < 24:
            group_ids.append(candidate)
            break
assert group_ids == ["Robin_education_Margarito", "Robin_lodging_Armand", "Robin_office_Sammie"]
group = electricity.loc["2017-02-06":"2017-02-12 23:00", group_ids]
assert len(group) == 168 and group.notna().all().all()
group.to_csv(data_dir / "robin_group_electricity_2017_week.csv")
provenance = {
    "building_id": building, "site": "Robin (UCL)",
    "paper": "https://doi.org/10.1038/s41597-020-00712-x",
    "dataset": "https://github.com/buds-lab/building-data-genome-project-2",
    "archive": "https://zenodo.org/records/3887306", "licence": "CC BY-SA 4.0",
    "sources": {key: {"path": path.relative_to(source_root).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for key, path in source_paths.items()},
    "extract": {"rows": len(extract), "start": str(grid.min()), "end": str(grid.max()),
                "missing": extract.isna().sum().to_dict(),
                "units": {"electricity_kwh": "kWh per hourly interval",
                          "hotwater_kwh": "kWh thermal energy per hourly interval",
                          "air_temperature_c": "degrees Celsius, observed"},
                "changes": "Select named columns and align to the supplied hourly grid. No filling or unit conversion."},
    "usage_profiles": {"start": period_start, "end": period_end,
                       "building_counts": profiles.groupby("usage_type").building_id.nunique().to_dict(),
                       "excluded_buildings": sorted(set(profile_buildings.building_id) - set(profiles.building_id)),
                       "missing_training_readings": int(electricity.loc[period_start:period_end, profile_buildings.building_id].isna().sum().sum()),
                       "flagged_flat_hours": int(profile_quality.flagged_flat_hours.sum()),
                       "flat_screening": "Exclude exact constant runs of at least 24 hours, identified within training only. Flags do not prove imputation.",
                       "normalisation": "Each building divided by its own maximum retained training electricity reading.",
                       "group_mean": "Equal-weight average of building mean profiles, computed in notebook.",
                       "zero_is_not_missing": True, "gap_filling": "none"},
    "selection": {"basis": "Complete positive 2017 electricity and no long identical-value runs; selected before model comparison.",
                  "full_2017_longest_flat_hours": int(selected_runs.max())},
    "aggregation": {"buildings": group_ids, "start": "2017-02-06 00:00", "end": "2017-02-12 23:00",
                    "rows": 168, "units": "kWh per one-hour interval; divide by 1 h for average kW",
                    "selection": "Margarito plus alphabetically first residence and office with complete 2017 training readings and no exact constant run >=24 h. Selection precedes aggregate/forecast comparison.",
                    "clock": "Same supplied Robin timestamps. No timezone conversion.",
                    "membership": "Fixed selected group; no documented shared feeder.",
                    "changes": "Select original cleaned source columns and complete training week. No filling, normalisation or zero substitution."},
    "artefacts": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in
                  ["robin_education_margarito_2017.csv", "robin_building_metadata.csv", "robin_usage_profiles.csv",
                   "building_quality_comparison.csv", "robin_profile_quality.csv", "robin_group_electricity_2017_week.csv"]},
    "versions": {"pandas": pd.__version__, "numpy": np.__version__},
}
(data_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
if args.data_only or not args.legacy_notebooks:
    print("Prepared 2017 data only; edited notebooks preserved.")
    raise SystemExit(0)

usage_intro = r"""### Compare profiles by building usage type

**Predict:** which usage type will show the largest weekday/weekend difference?

Compare 49 Robin-site buildings: 10 lodging/residential, 22 education and 17 office.
The supplied compact table contains each building's mean electricity at each hour for
weekdays and weekends, using observed readings from the training period only.
Two missing readings are excluded from their means; no values are imputed.

To compare shapes rather than building sizes, divide each building's readings by its
maximum observed training reading. Grey lines show individual buildings. The black
line gives each building equal weight, irrespective of floor area or number of readings.
This is a normalised profile, not a comparison of absolute energy use or efficiency.

After plotting, identify one weekday/weekend difference and one source of variation
within a usage type. Explain why the group mean alone is insufficient for forecasting
an individual building."""
usage_plot = r"""usage_profiles = pd.read_csv(PROFILES_PATH)
usage_metadata = pd.read_csv(ROOT / "data/week4_day1/robin/robin_building_metadata.csv")
profile_counts = usage_profiles.groupby("usage_type").building_id.nunique()
display(profile_counts.rename("Buildings"))
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
export_figure(fig, 19, "usage_type_profiles", "L2", "8 January–30 September 2016",
              "dimensionless; each building's training maximum = 1",
              "49 individual-building profiles (grey) and equal-weight usage means (black)", 49)
usage_means = usage_profiles.groupby(["usage_type", "day_type", "hour"]).normalised_mean.mean()
display(usage_means.groupby(["usage_type", "day_type"]).max().unstack().round(3))"""

def update_2017_cell(cell):
    """Authoring-only edits shared by the rebuilt pair and existing edited notebooks."""
    cell.source = cell.source.replace("Robin_education_Cecilia", "Robin_education_Margarito")
    cell.source = cell.source.replace("robin_education_cecilia.csv", "robin_education_margarito_2017.csv")
    cell.source = cell.source.replace("2016–2017", "2017").replace("2016", "2017")
    cell.source = cell.source.replace("2017-02-08", "2017-02-06").replace("2017-02-14", "2017-02-12")
    cell.source = cell.source.replace("8–14 February 2017", "6–12 February 2017")
    cell.source = cell.source.replace("8 February 2017", "6 February 2017")
    if cell.id == "overview":
        cell.source += ("\n\n**Forecasting dates:** training targets 8 January–30 September 2017; "
                        "test targets 1–28 October 2017. Issue forecasts at 23:00 for the following "
                        "day's 24 hours. No 2016 data are used for fitting or scoring.")
    elif cell.id == "l0":
        cell.source += ("\n\nA complete index does not guarantee reliable readings. This building's 2017 "
                        "electricity has no missing values and no constant runs longer than two hours. "
                        "The earlier Cecilia example contained long repeats in the source files; "
                        "we do not use it for this forecast experiment.")
    elif cell.id == "audit":
        anchor = 'display(audit)\nmeter_rows ='
        cell.source = cell.source.replace(anchor, '''electricity_run_lengths = original_load.groupby(original_load.ne(original_load.shift()).cumsum()).size()
audit.loc[len(audit)] = ["Longest constant electricity run (hours)", int(electricity_run_lengths.max())]
assert electricity_run_lengths.max() <= 2, "Recheck source quality before forecasting."
display(audit)
meter_rows =''')
    elif cell.id == "training_preparation":
        cell.source = '''### Check readings before forecasting

The selected 2017 electricity series has no missing readings or long constant
stretches. We use its values directly, without imputation.

There is one missing hot-water reading in training. The supplied plot illustrates
this gap. Hot-water energy and temperature are descriptive measurements, not forecast
inputs. We do not fill missing values in these columns or use them as scored targets.'''
    elif cell.id == "fill_training":
        start = cell.source.index("training_heat =")
        cell.source = '''history = original_load.loc[:TRAIN_END].copy()
assert history.notna().all()
load_for_features = original_load.loc[:TEST_END].copy()  # No imputation.
''' + cell.source[start:]
        cell.source = cell.source.replace('"28 March 2017 heat-meter gap; full-source audit"',
                                            'f"{gap:%d %B %Y} heat-meter gap; full-source audit"')
        cell.source = cell.source.replace('audit.iloc[[1, 2, 5, 6, 7, 8]]', 'audit.iloc[[1, 2, 5, 6, 8, 9]]')
    elif cell.id == "usage_type_intro":
        cell.source = cell.source.replace("Compare 49 Robin-site buildings: 10 lodging/residential, 22 education and 17 office.",
            "Compare the Robin-site residential, education and office buildings retained after\n"
            "data-quality screening. The count table reports the number in each category.")
        cell.source = cell.source.replace("Two missing readings are excluded from their means; no values are imputed.",
            "Source NaNs and exact constant stretches of at least 24 hours are excluded from\n"
            "these descriptive means. Such repeats may reflect earlier gap filling or a meter\n"
            "problem, but the cause is not known. The supplied table reports exclusions.\n"
            "Screening uses training data only; no source readings are overwritten or filled.")
        cell.source = cell.source.replace("maximum observed training reading", "maximum retained training reading")
    elif cell.id == "usage_type_plot":
        cell.source = cell.source.replace('"49 individual-building profiles (grey) and equal-weight usage means (black)", 49)',
            '"Retained individual-building profiles (grey) and equal-weight usage means (black)", int(usage_profiles.building_id.nunique()))')
        if 'profile_quality = pd.read_csv' not in cell.source:
            cell.source = cell.source.replace('display(profile_counts.rename("Buildings"))', '''display(profile_counts.rename("Buildings"))
profile_quality = pd.read_csv(ROOT / "data/week4_day1/robin/robin_profile_quality.csv")
display(profile_quality.groupby("usage_type")[["source_missing", "flagged_flat_hours", "retained_hours"]].sum())''')
    elif cell.id == "l4":
        cell.source = cell.source.replace("**8 January–30 September**", "**8 January–30 September 2017**")
        cell.source = cell.source.replace("**1–28 October**", "**1–28 October 2017**")
    elif cell.id == "optional_trace":
        cell.source = cell.source.replace("over two years", "over 2017")
    elif cell.id == "setup":
        cell.source = cell.source.replace("BDG2 Robin / UCL; 49 buildings", "BDG2 Robin / UCL; screened building profiles")
    return cell


for suffix in ["", "_solution"]:
    notebook = nbformat.read(root / f"tutorial_4_1a_building_load{suffix}.ipynb", as_version=4)
    for cell in notebook.cells:
        cell.source = cell.source.replace("Rat_education_Alfonso", building)
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
        if cell.id == "overview":
            cell.source = cell.source.replace("# Tutorial 4.1a:", "# Tutorial 4.1a (Robin / UCL alternative):")
            cell.source = cell.source.replace("a K–12 education building", "a college laboratory at UCL")
            cell.source = cell.source.replace("[Day 1 preparation](data/week4_day1/README.md)",
                                               "[Robin data preparation](data/week4_day1/robin/README.md)")
            cell.source = cell.source.replace("[original extract](data/tutorial_4_1/README.md)",
                                               "[selected building and profile data](data/week4_day1/robin/README.md)")
        elif cell.id == "l0":
            cell.source = cell.source[:cell.source.index("BDG2 provides only")]
            cell.source += ("BDG2 provides electricity and hot-water thermal-energy readings for this building. "
                            "The latter measures heat delivered through hot water, not water volume or gas use. "
                            "The plot below shows both meters separately. No gas meter is provided.")
        elif cell.id == "setup":
            cell.source = cell.source.replace('    fig.text(0.01, 0.01, textwrap.fill(',
                                               '    label = "BDG2 Robin / UCL; 49 buildings" if number == 19 else SOURCE_LABEL\n    fig.text(0.01, 0.01, textwrap.fill(')
            cell.source = cell.source.replace('f"{checkpoint} | {SOURCE_LABEL} |', 'f"{checkpoint} | {label} |')
            cell.source = cell.source.replace('ROOT / "figures" / "week4_day1"',
                                               'ROOT / "figures" / "week4_day1" / "robin"')
            cell.source = cell.source.replace('ROOT / "teaching" / "week4_day1_checkpoints"',
                                               'ROOT / "teaching" / "week4_day1_checkpoints" / "robin"')
            cell.source = cell.source.replace('"teaching" / "week4_day1_figure_manifest.csv"',
                                               '"teaching" / "week4_day1_robin_figure_manifest.csv"')
            cell.source = cell.source.replace('f"figures/week4_day1/{filename}"',
                                               'f"figures/week4_day1/robin/{filename}"')
            cell.source = cell.source.replace('"source_sha256": SOURCE_SHA256',
                                               '"source_sha256": hashlib.sha256(PROFILES_PATH.read_bytes()).hexdigest() if number == 19 else SOURCE_SHA256')
            cell.source = cell.source.replace('"tutorial_4_1a_building_load_solution.ipynb"',
                                               '"tutorial_4_1a_building_load_robin_solution.ipynb"')
            cell.source = cell.source.replace('ROOT / "data" / "tutorial_4_1" / "building_data_genome_sample.csv"',
                                               'ROOT / "data" / "week4_day1" / "robin" / "robin_education_cecilia.csv"')
            cell.source += ('\nPROFILES_PATH = ROOT / "data/week4_day1/robin/robin_usage_profiles.csv"'
                            '\nSLIDE_NUMBERS.update({18: 6, 19: 18})'
                            '\nSPEC_SLIDE_NUMBERS.update({18: 5, 19: 11})')
        elif cell.id == "audit":
            # Choose the first complete illustrated week with variation in both meters.
            # January's original display week has a constant electricity trace.
            cell.source = cell.source.replace('"2016-01-08":"2016-01-14 23:00"',
                                               '"2016-02-08":"2016-02-14 23:00"')
            cell.source = cell.source.replace('"2016-01-08 08:00":"2016-01-08 13:00"',
                                               '"2016-02-08 08:00":"2016-02-08 13:00"')
            cell.source = cell.source.replace('"8 January 2016, 08:00–13:00"',
                                               '"8 February 2016, 08:00–13:00"')
            cell.source = cell.source.replace('["Yes", "No", "No", "No", "No", "No", "No", "No"]',
                                               '["Yes", "No", "Yes", "No", "No", "No", "No", "No"]')
            start = cell.source.index('fig, ax = plt.subplots(figsize=(11, 3))')
            end = cell.source.index('audit = pd.DataFrame', start)
            cell.source = cell.source[:start] + r'''fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
for ax, column, label in zip(axes, ["electricity_kwh", "hotwater_kwh"],
                             ["Electricity", "Hot-water thermal energy"]):
    ax.plot(data.loc[meter_week.index].index, data.loc[meter_week.index, column], color="black")
    ax.set(ylabel="Hourly energy (kWh)", title=label)
axes[1].set_xlabel("Recorded date (Robin / UCL)")
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
export_figure(fig, 18, "additional_meters", "L0", "8–14 February 2016",
              "electricity and thermal kWh per hourly interval",
              "Separate measured electricity and hot-water energy for the same UCL building", len(meter_week))

''' + cell.source[end:]
            cell.source = cell.source.replace('display(audit)', '''audit = pd.concat([audit, pd.DataFrame({
    "Check": ["Stored hot-water NaNs", "Training hot-water NaNs"],
    "Count": [data.hotwater_kwh.isna().sum(), data.loc[TRAIN_START:TRAIN_END, "hotwater_kwh"].isna().sum()],
})], ignore_index=True)
display(audit)''')
            cell.source = cell.source.replace('["Recorded timestamp", "Hourly energy (kWh)", "Observed temperature (°C)"]',
                                               '["Recorded timestamp", "Electricity (kWh)", "Observed temperature (°C)", "Hot-water energy (kWh)"]')
            cell.source = cell.source.replace('widths=[0.36, 0.29, 0.35]', 'widths=[0.28, 0.21, 0.27, 0.24]')
        elif cell.id == "training_preparation":
            cell.source = r'''### Check missing readings before forecasting

Electricity is complete over the selected training and test periods, so no load
imputation is needed. The supplied preparation retains the previous-week replacement
and training-mean fallback for use if this building is later replaced by another one.
It leaves every observed reading unchanged.

Hot-water thermal energy has 12 missing training readings. The plot below illustrates
one of these gaps. We do not fill this descriptive series or use it as a forecasting
predictor. Temperature is also descriptive. The forecasting target remains electricity.'''
        elif cell.id == "fill_training":
            start = cell.source.index('gap_times =')
            cell.source = cell.source[:start] + r'''training_heat = data.loc[TRAIN_START:TRAIN_END, "hotwater_kwh"]
gap_times = training_heat.index[training_heat.isna()]
gap = gap_times[0]
zoom = pd.date_range(gap - pd.Timedelta(hours=12), gap + pd.Timedelta(hours=12), freq="h")
fig, axes = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [1.7, 1]})
axes[0].plot(zoom, data.loc[zoom, "hotwater_kwh"], color="black", marker="o", markersize=3, label="Observed thermal energy")
axes[0].axvspan(gap - pd.Timedelta(hours=1), gap + pd.Timedelta(hours=1), alpha=0.12, color="tab:red", label="Missing reading")
axes[0].set(title="A missing hot-water reading; electricity remains observed",
            xlabel="Recorded timestamp", ylabel="Hourly thermal energy (kWh)")
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
axes[0].legend(fontsize=9)
draw_table(axes[1], audit.iloc[[1, 2, 5, 6, 7, 8]], widths=[0.8, 0.2], fontsize=10)
export_figure(fig, 3, "quality_audit", "L0", "28 March 2016 heat-meter gap; full-source audit",
              "thermal kWh; missing counts", "Observed hot-water gap; no imputed forecast targets", len(zoom))'''
        elif cell.id == "calendar_plot":
            cell.source = cell.source.replace('table.iloc[:, 0] = selected.strftime',
                                               'table[table.columns[0]] = selected.strftime')
        elif cell.id == "worked_discussion":
            cell.source = cell.source[:cell.source.index('print("One reconstructed')]
            cell.source += ('print("Electricity is complete in the selected periods. Missing heat/weather readings are descriptive only; "'
                            ' "we do not use them as predictors. Instantaneous reading availability remains a teaching assumption.")')
    insert_at = next(i for i, cell in enumerate(notebook.cells) if cell.id == "l2")
    notebook.cells[insert_at:insert_at] = [
        nbformat.v4.new_markdown_cell(usage_intro, id="usage_type_intro"),
        nbformat.v4.new_code_cell(usage_plot, id="usage_type_plot", metadata={"tags": ["supplied"]}),
    ]
    if suffix == "_solution":
        notebook.cells.insert(insert_at + 2, nbformat.v4.new_code_cell(
            'print("Education and office profiles have stronger weekday daytime peaks; residential profiles peak later.")\n'
            'print("Buildings within each group differ in their baseline and peak levels. A group average cannot replace a building-specific forecast.")',
            id="usage_type_discussion", metadata={"tags": ["worked-discussion"]}))
    for cell in notebook.cells:
        update_2017_cell(cell)
    nbformat.validate(notebook)
    nbformat.write(notebook, root / f"tutorial_4_1a_building_load_robin{suffix}.ipynb")
print("Prepared Robin data and notebook pair. Original data and notebooks unchanged.")
