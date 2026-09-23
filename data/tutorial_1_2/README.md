# GEFCom2012 hourly load data

`load_data_full.csv` contains electricity-demand observations from the load-forecasting track of the Global Energy Forecasting Competition 2012 (GEFCom2012).

## Variables

- The first, unnamed column contains timestamps.
- `Z1` to `Z20` contain hourly demand for twenty geographical zones.
- `Z21` contains the aggregate demand across those zones and is the tutorial target.
- Demand values are recorded in MW.

## Temporal coverage and quality

- Coverage: 1 January 2005 00:00 to 31 December 2007 23:00.
- Nominal temporal resolution: hourly.
- Stored rows: 24,936.
- Duplicate timestamps: none.
- Explicit missing values in stored rows: none.
- Missing timestamps: 1,344, arranged as eight complete seven-day blocks.

Known missing timestamp blocks:

| Start | End | Duration |
|---|---|---:|
| 2005-03-06 00:00 | 2005-03-12 23:00 | 168 hours |
| 2005-06-20 00:00 | 2005-06-26 23:00 | 168 hours |
| 2005-09-10 00:00 | 2005-09-16 23:00 | 168 hours |
| 2005-12-25 00:00 | 2005-12-31 23:00 | 168 hours |
| 2006-02-13 00:00 | 2006-02-19 23:00 | 168 hours |
| 2006-05-25 00:00 | 2006-05-31 23:00 | 168 hours |
| 2006-08-02 00:00 | 2006-08-08 23:00 | 168 hours |
| 2006-11-22 00:00 | 2006-11-28 23:00 | 168 hours |

The notebook uses these gaps to illustrate time-index checks. It does not score imputation methods on them because their true values are unavailable. The 2007 portion is complete and supplies the artificial-gap and forecasting exercises.

## Provenance

This file is reused without modifying its values from:

`BENV0092/data processed/tutorial 3 - load forecasting/load_data_full.csv`

The BENV0092 course repository is available at https://github.com/akylasstrat/BENV0092.

Original source:

Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001
