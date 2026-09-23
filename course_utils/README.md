# Shared building-control utilities

`TwoStateReferencePlant` is the offline reference plant for Tutorials 6.1 to 8.2. It is a two-state $2R2C$-style model with ventilation, solar and internal gains, variable heat-pump COP, measurement noise and process noise. It is inspired by the BOPTEST `bestest_hydronic_heat_pump` case, but it is not an exact reduction of that Modelica model.

The student-facing $1R1C$ model is intentionally separate and simpler. Optimised actions are calculated with the $1R1C$ model and evaluated against the two-state plant.

## Replaceable plant interface

A replacement plant should provide:

```python
measurement = plant.reset(indoor_temperature_c, fabric_temperature_c)
result = plant.step(
    heat_pump_modulation,
    outdoor_temperature_c,
    solar_irradiance_w_m2,
    internal_gains_kw,
)
```

`step` should return at least `indoor_temperature_measured_c`, `heat_pump_electric_kw` and `delivered_heat_kw`. The simulation helpers accept a `plant_class` argument so that a later BOPTEST adapter can implement this interface without changing the tutorial algorithms.
