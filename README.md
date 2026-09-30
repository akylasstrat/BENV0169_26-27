# BENV0169: Data Analytics for Sustainable Buildings

Student tutorial material for BENV0169: Data Analytics for Sustainable Buildings, 2026/27.

## Available tutorial

### Tutorial 1.2 — Time-series forecasting

Open `tutorial_1_2_time_series_forecasting.ipynb` to work through data-quality checks, imputation and benchmark load forecasting. The solution notebook will be released after the tutorial.

Run the notebook from the repository root so that its relative data and image paths resolve correctly. The required data and provenance information are in `data/tutorial_1_2`. Other data directories support tutorials that will be released later and can be ignored for now.

Allow approximately 90 minutes to complete the main tutorial. The final exponential-smoothing exercise is optional.

## Set-up

The environment follows BENV0092 and uses Python 3.10.15.

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
