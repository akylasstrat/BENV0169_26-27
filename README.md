# BENV0169: Data Analytics for Sustainable Buildings

This repository contains teaching material for BENV0169. Tutorial 2 accompanies Lecture 2 on time-series data quality, missing-data treatment, benchmark forecasting and forecast evaluation.

## Tutorial 2 learning outcomes

After completing the tutorial, students should be able to:

- inspect a time index and distinguish absent timestamps from explicit `NaN` values;
- evaluate simple imputation methods using observations hidden from an otherwise complete period;
- construct and compare day-ahead, rolling-origin and fixed week-ahead benchmark forecasts without using future observations;
- calculate RMSE, MAPE and MASE safely;
- interpret how daily and weekly seasonality affect forecast performance.

The tutorial is designed for approximately 90 minutes.

## Notebooks

- `tutorial_2_time_series_forecasting.ipynb` is the student version. It contains numbered tasks, hints and `TODO` blocks.
- `tutorial_2_time_series_forecasting_solution.ipynb` is the completed version with executed tables, figures and brief answers to the interpretation questions.

Run either notebook from the repository root so that the repository-relative data path resolves correctly.

## Setup

Create and activate a virtual environment, then install the requirements. For example, on Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux, replace the activation command with:

```bash
source .venv/bin/activate
```

Launch JupyterLab from the repository root:

```bash
jupyter lab
```

## Data

The tutorial uses `Z21`, the aggregate hourly electricity-demand series in `data/tutorial_02/load_data_full.csv`. Columns `Z1` to `Z20` contain the twenty competition zones. See `data/tutorial_02/README.md` for coverage, units, known gaps and provenance.

Source: Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

The tutorial structure and coding style draw on the public [BENV0092 teaching repository](https://github.com/akylasstrat/BENV0092). This repository does not modify that reference project.
