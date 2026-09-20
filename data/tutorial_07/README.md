# Tutorial 7 data

This directory contains compact teaching extracts from two public energy datasets.

## `gefcom2012_system_load.csv`

- Hourly observations for 2007.
- `load_mw` is GEFCom2012 zone `Z21`, the aggregate of zones `Z1` to `Z20`.
- `air_temperature_c` is the mean across the 11 supplied weather stations, converted from °F to °C.
- The extract is used to compare seasonal-naïve, linear and tree-based day-ahead forecasts.

Source: Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

The source files were obtained from the BENV0092 teaching repository. The BENV0092 repository is not modified by this tutorial.

## `building_data_genome_sample.csv`

- Hourly observations for 2016 and 2017.
- `electricity_kwh` is the cleaned electricity-meter series for `Rat_education_Alfonso`.
- `air_temperature_c` is the air temperature for the corresponding `Rat` site.
- The building is a K–12 education building with a reported floor area of 7,900.4 m² and US/Eastern timezone.
- The hourly index is complete, but the source series contains 79 missing electricity readings and 11 missing weather values after alignment to the full index. These values are retained as `NaN`.

Source: Miller, C., Kathirgamanathan, A., Picchetti, B. et al. (2020) ‘The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition’, *Scientific Data*, 7, 368. https://doi.org/10.1038/s41597-020-00712-x

The Building Data Genome 2 repository distributes the data under the Creative Commons Attribution-ShareAlike 4.0 licence. This extract retains the source values, apart from selecting the building and weather columns and aligning them to an hourly index.

