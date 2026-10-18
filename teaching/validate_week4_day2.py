"""Focused instructor checks for the revised Tutorial 4.2 pair.

Execute the solution with nbconvert first. This validates notebook parity, metrics,
temporal availability and extension calculations. Student answers are inserted only
in memory; the distributed notebook is never filled by this script.
"""
from pathlib import Path
import argparse
import contextlib
import hashlib
import io
import json
import re
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
solution = nbformat.read(ROOT / "tutorial_4_2_forecast_uncertainty_solution.ipynb", as_version=4)
student = nbformat.read(ROOT / "tutorial_4_2_forecast_uncertainty.ipynb", as_version=4)
nbformat.validate(solution)
nbformat.validate(student)
solution_by_id = {cell.id: cell for cell in solution.cells}
common = [cell for cell in solution.cells
          if not {"solution-only", "worked-discussion"}.intersection(cell.metadata.get("tags", []))]
assert [cell.id for cell in common] == [cell.id for cell in student.cells]
exercise_count = 0
core_count = 0
for cell in student.cells:
    peer = solution_by_id[cell.id]
    assert cell.cell_type == peer.cell_type
    assert cell.metadata.get("tags", []) == peer.metadata.get("tags", [])
    if cell.cell_type == "code":
        assert not cell.outputs and cell.execution_count is None
        compile(cell.source, cell.id, "exec")
    if "exercise" in cell.metadata.get("tags", []):
        exercise_count += 1
        core_count += "extension" not in cell.metadata.get("tags", [])
        assert "# TODO" in cell.source and "NotImplementedError" in cell.source
        assert cell.source != peer.source
    else:
        assert cell.source == peer.source
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", cell.source)
    assert not any(term in cell.source for term in
                   ["RESTART_", "read_checkpoint(", "USE_SCORE_CHECKPOINTS", "MIN_CALIBRATION_COUNT"])
assert (core_count, exercise_count) == (5, 10)
execution_counts = [cell.execution_count for cell in solution.cells if cell.cell_type == "code"]
assert execution_counts == list(range(1, len(execution_counts)+1))
assert all(output.output_type != "error" for cell in solution.cells
           if cell.cell_type == "code" for output in cell.outputs)
# No unresolved lecturer annotations; mathematical intervals and links remain valid.
for cell in common:
    assert not re.search(r"\[(?:A bit|what does|fix wording|add an optional|too many|remove continuation|feels we|Earlier I)", cell.source, re.I)

# Run the student structure with the intended code answers inserted ONLY in memory.
namespace = {"__name__": "__validation__"}
plt.show = lambda: None
with contextlib.redirect_stdout(io.StringIO()):
    for cell in student.cells:
        if cell.cell_type == "code":
            source = solution_by_id[cell.id].source
            source = source.replace("EXPORT_ARTEFACTS = True", "EXPORT_ARTEFACTS = False")
            exec(compile(source, cell.id, "exec"), namespace)
plt.close("all")
n = namespace
cal, test = n["calibration"], n["test"]
assert [len(n["X_train"]), len(cal), len(test)] == [4176, 2208, 672]
assert n["building_source"].index.equals(pd.date_range("2017-01-01", "2017-12-31 23:00", freq="h"))
assert n["building_source"]["electricity_kwh"].notna().all()
np.testing.assert_allclose(n["load_scaler"].mean_, n["X_train"].mean(), rtol=0, atol=1e-10)
for frame in [cal, test]:
    pd.testing.assert_series_equal(frame["actual"], n["building_source"].loc[frame.index, "electricity_kwh"],
                                   check_names=False)
    assert frame.groupby("target_date").size().eq(24).all()
    assert frame.groupby("target_date")["origin_time"].nunique().eq(1).all()
    assert (frame["lag_24_time"] <= frame["origin_time"]).all()
    assert (frame["lag_168_time"] <= frame["origin_time"]).all()
    expected_lag24 = n["building_source"]["electricity_kwh"].shift(24).loc[frame.index]
    np.testing.assert_array_equal(n["X"].loc[frame.index, "load_lag_24"], expected_lag24)
np.testing.assert_allclose(n["pooled_offsets"], np.quantile(cal["error"], [.1, .9]))
for hour in range(24):
    expected = np.quantile(cal.loc[cal["hour"] == hour, "error"], [.1, .9])
    np.testing.assert_allclose(n["hourly_offsets"].loc[hour, ["lower_offset", "upper_offset"]], expected)
# Independent score calculation on the same actuals, including grouped scores.
for method, bounds in n["intervals"].items():
    for subset, mask in n["subsets"].items():
        a, l, u = test.loc[mask, "actual"].to_numpy(), bounds.loc[mask, "lower"].to_numpy(), bounds.loc[mask, "upper"].to_numpy()
        expected = [100*np.mean((l <= a) & (a <= u)), np.mean(u-l),
                    np.mean(u-l+10*np.maximum(l-a, 0)+10*np.maximum(a-u, 0))]
        row = n["score_table"].loc[(n["score_table"]["Method"] == method) &
                                   (n["score_table"]["Subset"] == subset)].iloc[0]
        np.testing.assert_allclose(row[["Coverage (%)", "Width (kWh)", "Winkler (kWh)"]].to_numpy(float), expected)
    for hour in [0, 10, 23]:
        mask = test["hour"].eq(hour)
        a = test.loc[mask, "actual"]
        selected = bounds.loc[mask]
        row = n["hourly_scores"].query("method == @method and hour == @hour").iloc[0]
        assert row["n"] == 28
        np.testing.assert_allclose(row["coverage"], 100*((selected["lower"] <= a) & (a <= selected["upper"])).mean())
coverage, width, winkler = n["coverage"], n["mean_width"], n["winkler"]
np.testing.assert_allclose(coverage([380, 450, 530], [400]*3, [500]*3), 100/3)
np.testing.assert_allclose(winkler([380, 450, 530], [400]*3, [500]*3), 800/3)
assert coverage([400, 500], [400, 400], [500, 500]) == 100
assert winkler([1], [1], [1]) == 0
index = pd.date_range("2020-01-01", periods=2, freq="h")
for metric in [coverage, width, winkler]:
    for inputs in [([1, 2], [0], [3, 3]), ([1, np.nan], [0, 0], [2, 2]),
                   ([1], [2], [1]), ([], [], []),
                   (pd.Series([1, 2], index=index), pd.Series([0, 0], index=index+pd.Timedelta(hours=1)),
                    pd.Series([3, 3], index=index))]:
        try:
            metric(*inputs)
        except ValueError:
            pass
        else:
            raise AssertionError("Metric accepted an invalid input")
for alpha in [0, 1, np.nan]:
    try:
        winkler([1], [0], [2], alpha)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid alpha accepted")
matrix = n["residual_matrix"]
assert matrix.shape == (92, 24) and matrix.notna().all().all()
np.testing.assert_allclose(n["ensemble_A_raw"], n["target_profile"][None, :] + matrix.to_numpy())
np.testing.assert_array_equal(np.sort(n["ensemble_A"], axis=0), np.sort(n["ensemble_B"], axis=0))
np.testing.assert_allclose(n["daily_energy_A"].mean(), n["daily_energy_B"].mean())
for smaller, larger in [(.5, .8), (.8, .95)]:
    assert (n["level_intervals"][smaller]["lower"] >= n["level_intervals"][larger]["lower"]-1e-12).all()
    assert (n["level_intervals"][smaller]["upper"] <= n["level_intervals"][larger]["upper"]+1e-12).all()
np.testing.assert_allclose(n["qr_bounds"]["lower"], np.maximum(0, np.minimum(n["q10_raw"], n["q90_raw"])))
np.testing.assert_allclose(n["qr_bounds"]["upper"], np.maximum(0, np.maximum(n["q10_raw"], n["q90_raw"])))
assert n["q10_model"].quantile == .1 and n["q90_model"].quantile == .9
np.testing.assert_allclose(n["joint_mean"], matrix.mean().to_numpy())
np.testing.assert_allclose(n["joint_covariance"], matrix.cov().to_numpy())
assert n["normal_scenarios"].shape == n["joint_scenarios"].shape == (500, 24)
for frame in [n["pv_train"], n["pv_cal"], n["pv_test"]]:
    assert (frame["weather_issue_time_utc"] <= frame["origin_time_utc"]).all()
    assert np.isfinite(frame[["actual", *n["PV_FEATURES"]]].to_numpy()).all()
assert [len(n["pv_train"]), len(n["pv_cal"]), len(n["pv_test"])] == [8758, 2184, 2208]
for frame in [n["pv_cal"], n["pv_test"]]:
    raw = n["pv_model"].predict(frame[n["PV_FEATURES"]])
    corrected = np.clip(raw, 0, 1)
    corrected[~frame["solar_available"].to_numpy()] = 0
    np.testing.assert_allclose(frame["corrected_forecast"], corrected)
for bounds in [n["pv_bounded"], n["pv_zero"], n["pv_hourly_bounds"]]:
    assert bounds.index.equals(n["pv_test"].index)
    assert ((bounds >= 0) & (bounds <= 1)).all().all()
    assert (bounds["lower"] <= bounds["upper"]).all()

# Confirm source files and unrelated notebooks remain unchanged from the recorded baseline.
baseline = json.loads((ROOT / "teaching/week4_day2_preflight_hashes.json").read_text(encoding="utf-8"))
unrelated_changes = []
for relative, digest in baseline.items():
    path = ROOT / relative
    current = hashlib.sha256(path.read_bytes()).hexdigest()
    if current != digest:
        if relative.startswith("data/"):
            raise AssertionError("Protected source changed: " + relative)
        unrelated_changes.append(relative)
manifest_path = ROOT / "teaching/week4_day2_figure_manifest.csv"
manifest = pd.read_csv(manifest_path)
assert len(manifest) == 46 and manifest["filename"].is_unique
assert set(manifest["figure_id"]) == {f"U{i:02}" for i in range(1, 20)} | {"E01", "E02", "E03"}
assert len(manifest.query("figure_id == 'U11'")) == 4
for _, row in manifest.iterrows():
    path = ROOT / row["filename"]
    assert path.exists()
    if path.suffix == ".pdf":
        content = path.read_bytes()
        assert content.startswith(b"%PDF") and b"/Subtype /Image" not in content
    else:
        with Image.open(path) as im:
            assert min(im.info.get("dpi", (0, 0))) >= 200
assert {str(path.relative_to(ROOT)).replace("\\", "/") for path in (ROOT / "figures/week4_day2").iterdir()} == set(manifest["filename"])
thumbnails = manifest.loc[manifest["filename"].str.endswith(".png")]
sheet = Image.new("RGB", (2400, 540*((len(thumbnails)+3)//4)), "white")
draw = ImageDraw.Draw(sheet)
for i, (_, row) in enumerate(thumbnails.iterrows()):
    x, y = (i % 4)*600, (i // 4)*540
    with Image.open(ROOT / row["filename"]) as im:
        im = im.convert("RGB")
        im.thumbnail((580, 495))
        sheet.paste(im, (x+(600-im.width)//2, y+30))
    draw.text((x+10, y+5), f"{row['figure_id']} {row['variant']}", fill="black")
sheet.save(ROOT / "teaching/week4_day2_contact_sheet.jpg", quality=90)
summary = {"status": "passed", "clean_kernel_execution": True, "paired_shared_cells": len(student.cells),
           "core_exercise_cells": core_count, "extension_exercise_cells": exercise_count-core_count,
           "student_outputs_empty": True, "unresolved_annotations": False,
           "temporal_availability_and_training_only_scaling": True,
           "independent_metrics_and_grouped_scores": True, "extension_calculations_checked": True,
           "checkpoint_dependencies_removed": True, "figure_exports": len(manifest),
           "visual_inspection_confirmed": args.visual_qa_confirmed,
           "unrelated_changes_preserved": unrelated_changes,
           "source_hashes": {"building": n["BUILDING_HASH"], "pv": n["PV_HASH"]},
           "building_scores": n["score_table"].to_dict("records"),
           "quantile_scores": n["qr_scores"].to_dict("records"),
           "normal_interval_scores": n["gaussian_interval_scores"].to_dict("records"),
           "gaussian_daily_energy": n["gaussian_energy"].agg(["mean", "std"]).to_dict(),
           "pv_hourly_scores": n["pv_hourly_scores"].to_dict("records"),
           "limitations": ["Summer-to-October error transfer", "28 test cases per target hour",
                           "One scenario target day", "Gaussian errors are an assumption",
                           "Forecast irradiance is an imperfect proxy for night"]}
for path in [ROOT / "teaching/week4_day2_validation_summary.json", ROOT / "data/week4_day2/validation_summary.json"]:
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
manifest["validation_status"] = "numerically and visually checked" if args.visual_qa_confirmed else "numerically checked; visual review pending"
manifest.to_csv(manifest_path, index=False)
print(f"Passed: {core_count} core and {exercise_count-core_count} extension exercise cells; {len(manifest)} figure exports.")
