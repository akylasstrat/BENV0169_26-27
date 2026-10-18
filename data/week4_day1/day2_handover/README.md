# Day 2 PV starting bundle

The prediction CSVs and metadata are generated locally and excluded from Git. Run `rebuild_cell.py` with export enabled before using this optional bundle. The current Tutorial 4.2 notebook rebuilds its own predictions and does not require it.

Release this bundle **at the start of Day 2**. It uses tutorial GEFCom2014 Zone 3, not assessment data. Keep it with instructor material before that release if students should complete the Day 1 fitting exercise unaided.

- `calibration_predictions.csv`: 2,184 aligned April–June 2013 targets.
- `test_predictions.csv`: 2,208 aligned July–September 2013 targets.
- `metadata.json`: fixed model/configuration, input order, split dates, source SHA256, versions, seed, fitted coefficients, correction rules and prediction-file checksums.
- `rebuild_cell.py`: transparent standalone cells which read the unchanged source, construct lag 1, fit the prescribed linear model on training only, predict calibration/test and apply corrections. No pickle or prior notebook state is required.

Each CSV contains `target_time_utc`, `origin_time_utc`, `weather_issue_time_utc`, `solar_available`, `actual`, `raw_forecast`, `corrected_forecast`. PV values are p.u.; timestamps are UTC. The proxy is a Boolean derived from forecast irradiance, not actual PV. Origin = target −1 hour; issue ≤ origin under the documented zero-release-latency assumption.

Run from the repository root in the existing environment:

```bash
python data/week4_day1/day2_handover/rebuild_cell.py
```

Alternatively, paste the file into one supplied cell in the Day 2 notebook and run it from the root. `EXPORT_ARTEFACTS=False` rebuilds the model and predictions without replacing CSV/metadata files. With export enabled, it recreates those files and their checksums. Do not refit using calibration or select the model using test scores. Day 2 calibration residuals are **actual − corrected_forecast**. This bundle intentionally contains no intervals, coverage, residual quantiles or probabilistic scores.

Read a bundle without rebuilding:

```python
from pathlib import Path
import pandas as pd

bundle_path = Path("data/week4_day1/day2_handover/calibration_predictions.csv")
calibration = pd.read_csv(bundle_path, index_col="target_time_utc")
calibration.index = pd.to_datetime(calibration.index, utc=True)
calibration["origin_time_utc"] = pd.to_datetime(calibration["origin_time_utc"], utc=True)
calibration["weather_issue_time_utc"] = pd.to_datetime(calibration["weather_issue_time_utc"], utc=True)
assert calibration.index.is_unique
assert calibration["weather_issue_time_utc"].le(calibration["origin_time_utc"]).all()
```

The source-target convention, single nominal-capacity exceedance, weather provenance and forecast-night limitation are documented in [the Day 1 data README](../README.md). In particular, do not alter actual outcomes when the forecast-night proxy is false.
