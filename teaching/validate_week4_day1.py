"""Instructor validation for the current Robin/UCL and PV tutorial pairs.

The notebooks contain only essential input guards. Repeated numerical checks live
here. TODO solutions are inserted only in memory; student files are not modified.
No assessment files or legacy notebook builders are opened.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--visual-qa-confirmed", action="store_true")
args = parser.parse_args()
# Full-precision scores recorded before the readability revision.
REFERENCE_SCORES = {
  "load": {
    "Weekly seasonal naïve": {
      "MAE (kWh)": 38.474404761904765,
      "RMSE (kWh)": 47.512069895072834,
      "n": 672
    },
    "Ridge": {
      "MAE (kWh)": 30.00090713697276,
      "RMSE (kWh)": 37.48657352180141,
      "n": 672
    },
    "Decision tree": {
      "MAE (kWh)": 30.89270793894088,
      "RMSE (kWh)": 40.06916927899716,
      "n": 672
    },
    "Random forest": {
      "MAE (kWh)": 29.12760176074489,
      "RMSE (kWh)": 37.51625105572565,
      "n": 672
    }
  },
  "pv": {
    "Persistence": {
      "MAE (p.u.)": 0.0591625792572463,
      "RMSE (p.u.)": 0.1101975234465058,
      "n": 2208
    },
    "Linear regression": {
      "MAE (p.u.)": 0.0496301964675839,
      "RMSE (p.u.)": 0.0929254200624036,
      "n": 2208
    },
    "Decision tree": {
      "MAE (p.u.)": 0.0482667005305537,
      "RMSE (p.u.)": 0.0963323281445906,
      "n": 2208
    },
    "Random forest": {
      "MAE (p.u.)": 0.0410008102305514,
      "RMSE (p.u.)": 0.0851558360443435,
      "n": 2208
    }
  }
}
states, report = {}, {}
plt.show = lambda: None
for part, label, exercises in [("a", "building_load", 8), ("b", "pv_point_forecasting", 6)]:
    name = f"tutorial_4_1{part}_{label}"
    student = nbformat.read(ROOT / (name + ".ipynb"), as_version=4)
    solution = nbformat.read(ROOT / (name + "_solution.ipynb"), as_version=4)
    nbformat.validate(student)
    nbformat.validate(solution)
    solved = {cell.id: cell for cell in solution.cells}
    common = [cell for cell in solution.cells if "worked-discussion" not in cell.metadata.get("tags", [])]
    assert [cell.id for cell in common] == [cell.id for cell in student.cells]
    count = 0
    for cell in student.cells:
        assert cell.cell_type == solved[cell.id].cell_type
        assert cell.metadata.get("tags", []) == solved[cell.id].metadata.get("tags", [])
        exercise = "exercise" in cell.metadata.get("tags", [])
        count += exercise
        if exercise:
            assert "# TODO" in cell.source and "NotImplementedError" in cell.source
            assert cell.source != solved[cell.id].source
        else:
            assert cell.source == solved[cell.id].source
        assert not any(term in cell.source for term in ["USE_LOAD_", "USE_P1_", "USE_P2_", "USE_P3_", "CHECKPOINT_DIR"])
        assert "C:/" not in cell.source and "C:" + chr(92) not in cell.source
        if cell.cell_type == "code":
            compile(cell.source, cell.id, "exec")
            assert cell.execution_count is None and not cell.outputs
    assert count == exercises
    counts = [cell.execution_count for cell in solution.cells if cell.cell_type == "code"]
    assert counts == list(range(1, len(counts)+1))
    assert not any(output.output_type == "error" for cell in solution.cells
                   if cell.cell_type == "code" for output in cell.outputs)
    namespace = {"display": lambda *args, **kwargs: None}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in student.cells:
            if cell.cell_type == "code":
                source = solved[cell.id].source.replace("EXPORT_ARTEFACTS = True", "EXPORT_ARTEFACTS = False")
                exec(compile(source, cell.id, "exec"), namespace)
    plt.close("all")
    states[part] = namespace
    report[part] = {"notebook": name, "exercises": count, "student_outputs_empty": True,
                    "paired_cells_match": True, "clean_kernel_execution": True,
                    "student_scaffolds_completed_only_in_memory": True}

load, pv = states["a"], states["b"]
assert len(load["original_load"]) == 8760
assert load["original_load"].notna().all()
assert load["electricity_run_lengths"].max() == 2
assert load["data"].hotwater_kwh.isna().sum() == 24
assert load["data"].air_temperature_c.isna().sum() == 7
pd.testing.assert_series_equal(load["original_load"], load["source"]["electricity_kwh"])
pd.testing.assert_series_equal(load["load_for_features"], load["original_load"].loc[:load["TEST_END"]])
assert [len(load["y_train"]), len(load["y_test"])] == [6384, 672]
assert load["X_train"].index.equals(load["y_train"].index)
assert load["X_test"].index.equals(load["y_test"].index)
assert load["y_train"].index.max() < load["y_test"].index.min()
np.testing.assert_allclose(load["daily_energy"], load["resolution_hourly"].to_numpy().reshape(7, 24).sum(axis=1))
np.testing.assert_allclose(load["scaler"].mean_, load["X_train"][load["RIDGE_COLUMNS"]].mean())
for lag in [24, 168]:
    targets = load["X_test"].index
    source_times = targets-pd.Timedelta(hours=lag)
    assert (source_times <= load["test_origins"]).all()
    np.testing.assert_allclose(load["X_test"][f"load_lag_{lag}"], load["original_load"].loc[source_times])
assert list(load["tree_model"].feature_names_in_) == list(load["forest_model"].feature_names_in_) == load["TREE_COLUMNS"]
np.testing.assert_allclose(load["forecasts"]["Weekly seasonal naïve"], load["X_test"]["load_lag_168"])
np.testing.assert_allclose(load["load_factor"], load["training_original"].mean()/load["training_original"].max())
overnight = load["training_original"].index.hour < 6
np.testing.assert_allclose(load["overnight_share"], load["training_original"].loc[overnight].sum()/load["training_original"].sum())
group = load["group_energy"]
assert group.shape == (168, 3) and group.notna().all().all()
assert group.index.equals(load["resolution_hourly"].index)
np.testing.assert_allclose(load["aggregate_power"], group.sum(axis=1))
np.testing.assert_allclose(load["aggregate_peak"], group.sum(axis=1).max())
assert load["aggregate_peak"] <= load["sum_individual_peaks"]
for row in load["daily_peak_scores"].itertuples():
    targets = pd.date_range(row.day, periods=24, freq="h")
    a, f = load["y_test"].loc[targets], load["forecasts"].loc[targets, row.model]
    np.testing.assert_allclose(row.mae_kwh, (a-f).abs().mean())
    np.testing.assert_allclose(row.peak_error_kw, abs(a.max()-f.max()))
profiles = load["usage_profiles"]
assert profiles.groupby("usage_type").building_id.nunique().to_dict() == {
    "Education": 22, "Lodging/residential": 10, "Office": 16}
np.testing.assert_allclose(profiles.normalised_mean, profiles.mean_electricity_kwh/profiles.training_max_kwh)
for state, table_key, actual_key, forecasts_key, unit, ref_key in [
    (load, "comparison", "y_test", "forecasts", "kWh", "load"),
    (pv, "pv_comparison", "y_pv_test", "pv_forecasts", "p.u.", "pv")]:
    a, forecasts = state[actual_key], state[forecasts_key]
    assert forecasts.index.equals(a.index)
    assert np.isfinite(forecasts).all().all() and forecasts.ge(0).all().all()
    for model in forecasts:
        error = a.to_numpy()-forecasts[model].to_numpy()
        expected = [np.abs(error).mean(), np.sqrt(np.mean(error**2))]
        np.testing.assert_allclose(state[table_key].loc[model, [f"MAE ({unit})", f"RMSE ({unit})"]], expected)
        baseline = REFERENCE_SCORES[ref_key][model]
        np.testing.assert_allclose(expected, [baseline[f"MAE ({unit})"], baseline[f"RMSE ({unit})"]],
                                   rtol=0, atol=1e-10)
    index = pd.date_range("2020-01-01", periods=3, freq="h")
    actual = pd.Series([2., 4., 8.], index=index)
    forecast = pd.Series([1., 5., 6.], index=index)
    np.testing.assert_allclose([state["mae"](actual, forecast), state["rmse"](actual, forecast)],
                               [4/3, np.sqrt(2)])
    for metric in [state["mae"], state["rmse"]]:
        for bad in [forecast.iloc[:2], forecast.set_axis(index+pd.Timedelta(hours=1)),
                    pd.Series([1., np.nan, 6.], index=index)]:
            try:
                metric(actual, bad)
            except ValueError:
                pass
            else:
                raise AssertionError("Metric accepted invalid alignment or non-finite values")

assert [len(pv["X_pv_train"]), len(pv["X_pv_calibration"]), len(pv["X_pv_test"])] == [8758, 2184, 2208]
assert pv["pv_source"].index.is_unique
assert pv["pv_source"].index.equals(pd.date_range(pv["pv_source"].index.min(), pv["pv_source"].index.max(), freq="h"))
pd.testing.assert_series_equal(pv["pv"]["solar_power_pu_raw"], pv["pv_source"]["solar_power_pu"], check_names=False)
assert len(pv["changed_targets"]) == 1
assert (pv["pv"].forecast_issue_time_utc <= pv["pv"].origin_time_utc).all()
assert list(pv["pv_tree"].feature_names_in_) == list(pv["pv_forest"].feature_names_in_) == pv["PV_FEATURE_COLUMNS"]
for inputs, targets in [(pv["X_pv_calibration"], pv["calibration_index"]), (pv["X_pv_test"], pv["test_index"])]:
    source_times = targets-pd.Timedelta(hours=1)
    np.testing.assert_allclose(inputs["pv_lag_1"], pv["pv"].loc[source_times, "solar_power_pu"])
    assert not targets.isin(pv["changed_targets"].index).any()
for model, prefix in [(pv["pv_linear_model"], "linear"), (pv["pv_tree"], "tree"), (pv["pv_forest"], "forest")]:
    for period, inputs, targets in [
        ("calibration", pv["X_pv_calibration"], pv["calibration_index"]),
        ("test", pv["X_pv_test"], pv["test_index"])]:
        raw = model.predict(inputs[pv["PV_FEATURE_COLUMNS"]])
        np.testing.assert_allclose(raw, pv[f"{prefix}_{period}_raw"])
        corrected = np.clip(raw, 0, 1)
        corrected[~pv["pv"].loc[targets, "solar_available"].to_numpy()] = 0
        np.testing.assert_allclose(corrected, pv[f"{prefix}_{period}"])
assert pv["pv_forecasts"].le(1).all().all()
assert pv["pv_forecasts"].loc[~pv["solar_subset"]].eq(0).all().all()
assert int(pv["solar_subset"].sum()) == 1044
for subset, model, score_mae, score_rmse, count in pv["pv_subset_comparison"].itertuples(index=False, name=None):
    mask = pd.Series(True, index=pv["y_pv_test"].index) if subset == "All hours" else pv["solar_subset"]
    error = pv["y_pv_test"].loc[mask]-pv["pv_forecasts"].loc[mask, model]
    assert count == len(error)
    np.testing.assert_allclose([score_mae, score_rmse], [error.abs().mean(), np.sqrt((error**2).mean())])
# The review does not modify data, Day 2, Week 1 or the older handover bundle.
protected_files = json.loads((ROOT / "teaching/week4_day1_review_protected_hashes.json").read_text())
for relative, expected in protected_files.items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative

robin_manifest = pd.read_csv(ROOT / "teaching/week4_day1_robin_figure_manifest.csv")
pv_manifest = pd.read_csv(ROOT / "teaching/week4_day1_figure_manifest.csv")
pv_manifest = pv_manifest.loc[pv_manifest.figure_id.isin([f"F{i:02}" for i in range(12, 17)])]
manifest = pd.concat([robin_manifest, pv_manifest], ignore_index=True).sort_values("figure_id")
assert len(manifest) == 22 and manifest.figure_id.is_unique
assert set(manifest.notebook) == {"tutorial_4_1a_building_load_solution.ipynb",
                                  "tutorial_4_1b_pv_point_forecasting_solution.ipynb"}
for row in manifest.itertuples():
    path = ROOT / row.filename
    assert path.exists() and path.with_suffix(".svg").exists()
    assert "<svg" in path.with_suffix(".svg").read_text(encoding="utf-8")
    with Image.open(path) as im:
        assert min(im.info["dpi"]) >= 200
manifest["status"] = "numerically and visually checked" if args.visual_qa_confirmed else "numerically checked; visual review pending"
manifest.to_csv(ROOT / "teaching/week4_day1_lecture_figure_manifest.csv", index=False)
sheet = Image.new("RGB", (2400, 6*540), "white")
draw = ImageDraw.Draw(sheet)
for i, row in enumerate(manifest.itertuples()):
    x, y = i%4*600, i//4*540
    with Image.open(ROOT / row.filename) as im:
        im = im.convert("RGB")
        im.thumbnail((580, 490))
        sheet.paste(im, (x+(600-im.width)//2, y+35))
    draw.text((x+10, y+10), row.figure_id, fill="black")
sheet.save(ROOT / "teaching/week4_day1_contact_sheet.jpg", quality=90)
report.update({"source_and_unrelated_notebook_hashes_unchanged": True, "prior_scores_unchanged": True,
               "core_exercise_cells": 14, "png_svg_figure_pairs": 22,
               "visual_inspection_confirmed": args.visual_qa_confirmed,
               "building_scores": load["comparison"].reset_index().to_dict("records"),
               "pv_scores": pv["pv_comparison"].reset_index().to_dict("records"),
               "model_settings_and_feature_encodings_unchanged": True,
               "checkpoint_dependencies_removed": True,
               "training_only_scaling_and_temporal_availability": "passed"})
(ROOT / "teaching/week4_day1_validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("Passed: 14 exercise cells; prior scores unchanged; temporal and metric checks; 22 PNG/SVG pairs.")

