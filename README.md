# BENV0169: Data Analytics for Sustainable Buildings

Teaching material for BENV0169: Data Analytics for Sustainable Buildings, 2026/27.

## Set-up

The environment follows BENV0092 and uses Python 3.10.15.

```bash
conda create -n BENV0169 python=3.10.15 ipython
conda activate BENV0169
pip install -r requirements.txt
```

## Tutorial 2

- `tutorial_2_time_series_forecasting.ipynb`: student notebook
- `tutorial_2_time_series_forecasting_solution.ipynb`: solution notebook

Run the notebooks from the repository root. Tutorial data and provenance are in `data/tutorial_02`.

Data source: Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

## Tutorial 7

- `tutorial_7_smart_meter_load_forecasting.ipynb`: student notebook
- `tutorial_7_smart_meter_load_forecasting_solution.ipynb`: solution notebook

The tutorial introduces smart-meter exploration and simple machine-learning pipelines for day-ahead load forecasting. It uses GEFCom2012 aggregate demand and a Building Data Genome 2 education building. Data and provenance are in `data/tutorial_07`.

## Tutorial 8

- `tutorial_8_pv_probabilistic_forecasting.ipynb`: student notebook
- `tutorial_8_pv_probabilistic_forecasting_solution.ipynb`: solution notebook

The tutorial introduces residual prediction intervals, linear quantile regression and Gaussian trajectory scenarios. It reuses the GEFCom2012 data from Tutorial 7. The PV section will be completed after a suitable dataset is selected.
