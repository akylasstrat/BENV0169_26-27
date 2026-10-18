# BENV0169: Data Analytics for Sustainable Buildings

Student tutorial material for BENV0169: Data Analytics for Sustainable Buildings, 2026/27.

## Tutorial 1.2

The tutorial is split into two self-contained parts:

- [Data quality and imputation](tutorial_1_2_time_series_forecasting.ipynb): inspect missing data and compare imputation methods (about 45 minutes).
- [Introduction to day-ahead load forecasting](tutorial_1_2_forecast_intro.ipynb): construct naïve, seasonal-naïve and linear-regression forecasts, then evaluate and plot their errors (about 85 minutes).

Notebooks ending in `_solution` contain worked solutions and will be released after the tutorial.

Run the notebook from the repository root so that its relative data and image paths resolve correctly. The required data and provenance information are in `data/tutorial_1_2`.

## Week 4, Day 1

- [Building demand — Robin / UCL](tutorial_4_1a_building_load.ipynb): meter interpretation, demand profiles, aggregation and day-ahead forecasting exercises (70–90 minutes).
- [PV point forecasting](tutorial_4_1b_pv_point_forecasting.ipynb): one-hour-ahead forecasts (35–45 minutes).

Both notebooks are self-contained and use the existing environment. Worked notebooks end in `_solution` and are released separately. [Robin data](data/week4_day1/robin/README.md), [PV data preparation](data/week4_day1/README.md) and [instructor notes](teaching/week4_day1_instructor_notes.md) accompany them. Run from the root with `jupyter lab`; no previous notebook state or checkpoint files are required.

## Week 4, Day 2

[Forecast uncertainty](tutorial_4_2_forecast_uncertainty.ipynb) uses Robin/UCL building forecasts to compare calibration-error prediction intervals and daily scenario dependence, followed by a supplied PV example. Self-study extensions cover linear quantile regression, Gaussian scenarios and PV intervals by hour. The worked solution is released separately. The core session has 200 teaching minutes plus a separate 15–20-minute break; extensions are outside this timing. Run from the root with `jupyter lab`; no previous notebook state is needed.

[Data/method notes](data/week4_day2/README.md) and [instructor notes](teaching/week4_day2_instructor_notes.md) accompany the lesson.

## Generated files

Saved lecture plots, forecast tables, checkpoints, manifests, validation reports and teaching archives are local artifacts excluded from Git. Required input datasets, provenance, notebook illustrations and source scripts are versioned. Running a notebook may regenerate outputs; set `EXPORT_ARTEFACTS = False` where available to display results without exporting them.

## Set-up

The environment uses Python 3.10.15.

```bash
conda create -n BENV0169 python=3.10.15 ipython
conda activate BENV0169
pip install -r requirements.txt
```

Launch JupyterLab from the repository root:

```bash
jupyter lab
```

## Data source

The principal dataset contains hourly electricity-demand observations from the Global Energy Forecasting Competition 2012:

Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

Further provenance and data-quality information are provided in `data/tutorial_1_2/README.md`.
