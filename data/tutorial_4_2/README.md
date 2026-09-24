# Tutorial 4.2 data

Tutorial 4.2 uses two compact teaching extracts.

## GEFCom2012 aggregate load

The load exercises reuse `data/tutorial_4_1/gefcom2012_system_load.csv`, so the repository does not store a second copy. It contains hourly aggregate load (`load_mw`) and the mean historical weather forecast (`air_temperature_c`) for 2007. The aggregate load is GEFCom2012 zone `Z21`. The target-hour weather forecast has the same valid-time index as the load observation.

Source: Hong, T., Pinson, P. and Fan, S. (2014) ‘Global Energy Forecasting Competition 2012’, *International Journal of Forecasting*, 30(2), pp. 357–363. https://doi.org/10.1016/j.ijforecast.2013.07.001

## GEFCom2014 solar Zone 3

`gefcom2014_solar_zone3.csv` is a cleaned extract from the solar track of the Global Energy Forecasting Competition 2014.

- **Zone:** 3 of the three Australian solar zones. Exact plant locations were not disclosed by the competition organisers.
- **Period:** 1 April 2012 01:00 UTC to 1 July 2014 00:00 UTC. The source timestamps have no explicit offset; they are interpreted as UTC because daylight spans the evening and early-morning UTC hours and mean solar output peaks near 02:00 UTC, consistent with midday in eastern Australia.
- **Resolution:** hourly, with 19,704 unique and complete timestamps.
- **Target:** normalised solar power (`solar_power_pu`). Values are nominally on a 0–1 per-unit scale; the source maximum is 1.00355 because of normalisation or rounding.
- **Weather:** ECMWF numerical weather forecasts issued at 00:00 UTC, with lead hours 1–24 retained for every forecast day.

Zone 3 was selected because, among the three zones, it has the strongest same-timestamp correlation between normalised power and the derived surface-solar-radiation forecast. Selection is for a clear teaching example, not a claim that Zone 3 is intrinsically easier to forecast.

### Weather-variable mapping

The source uses ECMWF GRIB parameter numbers. They were identified from the official ECMWF parameter database and checked against their units and temporal patterns.

| Source field | Descriptive output column | Interpretation and unit |
|---|---|---|
| `VAR78` | `forecast_total_column_cloud_liquid_water_kg_m2` | Total column cloud liquid water, kg/m² |
| `VAR79` | `forecast_total_column_cloud_ice_water_kg_m2` | Total column cloud ice water, kg/m² |
| `VAR134` | `forecast_surface_pressure_pa` | Surface pressure, Pa |
| `VAR157` | `forecast_relative_humidity_pct` | Relative humidity, % |
| `VAR164` | `forecast_total_cloud_cover_fraction` | Total cloud cover, fraction from 0 to 1 |
| `VAR165` | `forecast_wind_u_10m_m_s` | 10 m eastward wind component, m/s |
| `VAR166` | `forecast_wind_v_10m_m_s` | 10 m northward wind component, m/s |
| `VAR167` | `forecast_air_temperature_c` | 2 m air temperature, converted from K to °C |
| `VAR169` | surface-solar-radiation columns | Surface solar radiation downwards, accumulated J/m² and derived W/m² |
| `VAR175` | surface-thermal-radiation columns | Surface thermal radiation downwards, accumulated J/m² and derived W/m² |
| `VAR178` | top-net-solar-radiation columns | Top net solar radiation, accumulated J/m² and derived W/m² |
| `VAR228` | precipitation columns | Total precipitation, accumulated m and derived hourly mm |
| `POWER` | `solar_power_pu` | Realised normalised solar power |

`forecast_wind_speed_10m_m_s` is derived from the two wind components. `zone_id`, `forecast_issue_time_utc` and `forecast_lead_hours` preserve the forecast-trajectory structure.

### Converting accumulated fields

`VAR169`, `VAR175`, `VAR178` and `VAR228` reset at the beginning of each daily forecast trajectory. They are differenced **within the same trajectory**, never across two forecast origins. At lead hour 1, the supplied accumulation is the increment since 00:00 UTC. Radiation-energy increments are divided by 3600 seconds to obtain hourly mean flux in W/m², while precipitation increments are converted from metres to millimetres.

Small negative differences caused by numerical precision are clipped to zero. The accumulated source columns remain in the extract, so the transformation is auditable.

The site coordinates are unavailable. `is_daylight_forecast` is therefore 1 when forecast surface solar radiation exceeds 1 W/m² and 0 otherwise. This operational proxy uses only information available in the weather forecast; it does not inspect realised PV power.

### Provenance

Source: Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A. and Hyndman, R.J. (2016) ‘Probabilistic energy forecasting: Global Energy Forecasting Competition 2014 and beyond’, *International Journal of Forecasting*, 32(3), pp. 896–913. https://doi.org/10.1016/j.ijforecast.2016.02.001

Variable definitions: ECMWF Parameter Database, https://codes.ecmwf.int/grib/param-db/

The reproducible extraction and validation steps are in `scripts/build_tutorial_4_2_pv_dataset.py`. The script accepts the downloaded source CSV as an argument and never embeds a user-specific path.
