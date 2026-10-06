# Assessment data

The four CSV files are the frozen data release for BENV0169 Assessment 1, Task 1.

- `building_load.csv`: hourly interval electricity use (`electricity_kwh`) and observed outdoor temperature (`temperature_c`) for the selected education building. The timestamp is the supplied local-clock teaching index; retain it as provided and do not perform a daylight-saving conversion. Q1 includes deliberately introduced duplicate timestamps, omitted hours and stored missing values.
- `imputation_reference.csv`: complete observed electricity data for the controlled Q2 segment. Use it only for scoring and plotting.
- `imputation_gapped.csv`: an independently corrupted copy of the Q2 segment. Detect missing periods from this file and pass only this series to reconstruction functions.
- `pv_forecasts.csv`: capacity-normalised PV observations and weather forecasts for GEFCom2014 Solar Zone 3. Valid and issue timestamps are UTC. Surface solar radiation is hourly mean forecast irradiance in W/m², derived from the source accumulated radiation within each weather-forecast trajectory. `forecast_lead_hours` is the weather-run lead, not the one-hour PV forecasting horizon.

Building data derive from the Building Data Genome Project 2. PV and weather forecasts derive from GEFCom2014. The assessment ZIP is the authoritative release; do not substitute later downloads.
