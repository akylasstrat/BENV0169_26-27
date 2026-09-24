# BENV0169: Data Analytics for Sustainable Buildings

Teaching material for BENV0169: Data Analytics for Sustainable Buildings, 2026/27.

## Set-up

The environment follows BENV0092 and uses Python 3.10.15.

```bash
conda create -n BENV0169 python=3.10.15 ipython
conda activate BENV0169
pip install -r requirements.txt
```

## Tutorial 1.2

- `tutorial_1_2_time_series_forecasting.ipynb`: student notebook
- `tutorial_1_2_time_series_forecasting_solution.ipynb`: solution notebook

Run the notebooks from the repository root. Tutorial data and provenance are in `data/tutorial_1_2`.

Data source: Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

## Tutorial 4.1

- `tutorial_4_1_smart_meter_load_forecasting.ipynb`: student notebook
- `tutorial_4_1_smart_meter_load_forecasting_solution.ipynb`: solution notebook

The tutorial introduces smart-meter exploration and simple machine-learning pipelines for day-ahead load forecasting. It uses GEFCom2012 aggregate demand and a Building Data Genome 2 education building. Data and provenance are in `data/tutorial_4_1`.

## Tutorial 4.2

- `tutorial_4_2_pv_probabilistic_forecasting.ipynb`: student notebook
- `tutorial_4_2_pv_probabilistic_forecasting_solution.ipynb`: solution notebook

The tutorial introduces residual prediction intervals, linear quantile regression and Gaussian trajectory scenarios using GEFCom2012 load, then transfers the point and probabilistic methods to a physically constrained GEFCom2014 solar example.

## Building control tutorials

Tutorials 6.1 to 8.2 cover thermal simulation, RC model identification, optimisation, flexible assets, model predictive control and controller evaluation. Each tutorial has a student notebook and a matching `_solution` notebook.

The shared data are in `data/reference_building`, and the replaceable reference plant is in `course_utils`. Tutorial 6.2 also includes an optional occupied-house measurement exercise using the data in `data/tutorial_6_2`. Optional instructor material is in `instructor`.
