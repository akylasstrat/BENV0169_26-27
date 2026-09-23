# Reference building data

These files support Tutorials 6.1 to 8.2. They describe a synthetic single-zone dwelling driven by a two-temperature reference plant in `course_utils/reference_building.py`.

The case is inspired by the open BOPTEST `bestest_hydronic_heat_pump` dwelling: a 192 m² single-zone residential building with an air-to-water heat pump and hydronic floor heating. The teaching model is not an exact reduction or validated reproduction of the BOPTEST model. Its parameters were selected to give plausible slow fabric dynamics, heat-pump operation and deliberate mismatch with a fitted $1R1C$ controller model.

## Files

- `identification_data.csv`: 28 days of half-hourly simulated measurements under a varied heating signal. Indoor-temperature measurement noise and process noise are included.
- `operation_conditions.csv`: 14 days of realised weather, gains, occupancy, comfort limits, base load, PV availability and tariffs.
- `operation_forecasts.csv`: pre-generated, origin-specific weather and PV forecasts in long form. Each row records an issue time, valid time and lead step.
- `verified_controller_model.json`: common $1R1C$ model for the optimisation and MPC tutorials.
- `reference_plant_parameters.json`: parameters of the teaching reference plant.

All timestamps mark the start of a 30-minute interval. Temperatures use degrees Celsius, power uses kW, irradiance uses W/m², energy prices use £/kWh and the heat-pump control is a dimensionless modulation between zero and one.

The data are deterministic and can be regenerated with:

```bash
python scripts/generate_reference_building_data.py
```

Reference inspiration: IBPSA BOPTEST, `bestest_hydronic_heat_pump`, https://ibpsa.github.io/project1-boptest/docs-testcases/bestest_hydronic_heat_pump/
