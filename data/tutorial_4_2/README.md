# Tutorial 4.2 data

The probabilistic forecasting exercises reuse `data/tutorial_4_1/gefcom2012_system_load.csv`. This avoids storing a second copy of the same GEFCom2012 extract.

The file contains hourly aggregate load (`load_mw`) and the mean historical weather forecast (`air_temperature_c`) for 2007. The aggregate load corresponds to GEFCom2012 zone `Z21`. The weather forecast has the same valid-time index as the load observation and is used directly as an origin-valid predictor in Tutorial 4.2.

Source: Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

The PV dataset has not yet been selected. When it is added, this README should document its source, licence, site location, timezone, units, sampling convention, installed capacity, weather variables and known data-quality issues.
