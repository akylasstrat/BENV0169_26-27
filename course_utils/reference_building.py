"""A small, replaceable reference plant for the Weeks 6--8 tutorials.

The model is a teaching surrogate inspired by the open BOPTEST
``bestest_hydronic_heat_pump`` dwelling.  It is not an exact reduction of
that Modelica model.  The public interface is intentionally small so that a
future BOPTEST adapter can expose the same ``reset`` and ``step`` methods.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ReferenceBuildingParameters:
    """Parameters of the two-state reference building."""

    resistance_outdoor_fabric_k_per_kw: float = 6.0
    resistance_air_fabric_k_per_kw: float = 0.8
    capacitance_air_kwh_per_k: float = 0.8
    capacitance_fabric_kwh_per_k: float = 18.0
    ventilation_conductance_kw_per_k: float = 0.08
    solar_aperture_kw_per_w_m2: float = 0.010
    gains_to_air_fraction: float = 0.45
    heating_to_air_fraction: float = 0.25
    heat_pump_max_electric_kw: float = 5.0
    nominal_cop: float = 3.0
    cop_reference_temperature_c: float = 5.0
    cop_slope_per_k: float = 0.05
    minimum_cop: float = 2.0
    maximum_cop: float = 4.2

    def to_dict(self):
        return asdict(self)


class TwoStateReferencePlant:
    """Two-temperature reference plant with a heat-pump actuator.

    Parameters
    ----------
    parameters:
        Physical and actuator parameters.
    step_hours:
        External communication interval in hours.
    integration_substeps:
        Euler substeps used internally during each communication interval.
    measurement_noise_std_c:
        Standard deviation of indoor-temperature measurement noise.
    process_noise_std_c:
        Standard deviation of process noise added to each state per interval.
    random_state:
        Seed for reproducible noise.
    """

    def __init__(
        self,
        parameters=None,
        step_hours=0.5,
        integration_substeps=6,
        measurement_noise_std_c=0.05,
        process_noise_std_c=0.01,
        random_state=42,
    ):
        self.parameters = parameters or ReferenceBuildingParameters()
        self.step_hours = float(step_hours)
        self.integration_substeps = int(integration_substeps)
        self.measurement_noise_std_c = float(measurement_noise_std_c)
        self.process_noise_std_c = float(process_noise_std_c)
        self.rng = np.random.default_rng(random_state)
        self.indoor_temperature_c = np.nan
        self.fabric_temperature_c = np.nan

    def reset(self, indoor_temperature_c=20.0, fabric_temperature_c=20.0):
        """Reset the hidden plant state and return the first measurement."""

        self.indoor_temperature_c = float(indoor_temperature_c)
        self.fabric_temperature_c = float(fabric_temperature_c)
        return self.observe()

    def observe(self):
        """Return a noisy indoor-temperature measurement."""

        noise = self.rng.normal(0.0, self.measurement_noise_std_c)
        return float(self.indoor_temperature_c + noise)

    def actuation(self, heat_pump_modulation, outdoor_temperature_c):
        """Map a modulation command in ``[0, 1]`` to electrical and thermal power."""

        p = self.parameters
        modulation = float(np.clip(heat_pump_modulation, 0.0, 1.0))
        cop = p.nominal_cop + p.cop_slope_per_k * (
            float(outdoor_temperature_c) - p.cop_reference_temperature_c
        )
        cop = float(np.clip(cop, p.minimum_cop, p.maximum_cop))
        electric_power_kw = modulation * p.heat_pump_max_electric_kw
        delivered_heat_kw = electric_power_kw * cop
        return electric_power_kw, delivered_heat_kw, cop

    def step(
        self,
        heat_pump_modulation,
        outdoor_temperature_c,
        solar_irradiance_w_m2=0.0,
        internal_gains_kw=0.0,
    ):
        """Advance the plant by one communication interval.

        Returns a dictionary containing the new measurement, hidden states and
        realised equipment power.  Controllers should use only the measurement
        and the quantities that would be operationally observable.
        """

        p = self.parameters
        electric_kw, delivered_heat_kw, cop = self.actuation(
            heat_pump_modulation, outdoor_temperature_c
        )
        solar_gain_kw = max(0.0, float(solar_irradiance_w_m2)) * p.solar_aperture_kw_per_w_m2
        total_gains_kw = solar_gain_kw + max(0.0, float(internal_gains_kw))

        dt = self.step_hours / self.integration_substeps
        for _ in range(self.integration_substeps):
            air_to_fabric_kw = (
                self.fabric_temperature_c - self.indoor_temperature_c
            ) / p.resistance_air_fabric_k_per_kw
            outdoor_to_fabric_kw = (
                float(outdoor_temperature_c) - self.fabric_temperature_c
            ) / p.resistance_outdoor_fabric_k_per_kw
            ventilation_kw = p.ventilation_conductance_kw_per_k * (
                float(outdoor_temperature_c) - self.indoor_temperature_c
            )

            air_heat_kw = (
                air_to_fabric_kw
                + ventilation_kw
                + p.gains_to_air_fraction * total_gains_kw
                + p.heating_to_air_fraction * delivered_heat_kw
            )
            fabric_heat_kw = (
                -air_to_fabric_kw
                + outdoor_to_fabric_kw
                + (1.0 - p.gains_to_air_fraction) * total_gains_kw
                + (1.0 - p.heating_to_air_fraction) * delivered_heat_kw
            )

            self.indoor_temperature_c += dt * air_heat_kw / p.capacitance_air_kwh_per_k
            self.fabric_temperature_c += dt * fabric_heat_kw / p.capacitance_fabric_kwh_per_k

        if self.process_noise_std_c > 0:
            self.indoor_temperature_c += self.rng.normal(0.0, self.process_noise_std_c)
            self.fabric_temperature_c += self.rng.normal(0.0, self.process_noise_std_c / 3.0)

        return {
            "indoor_temperature_measured_c": self.observe(),
            "indoor_temperature_true_c": float(self.indoor_temperature_c),
            "fabric_temperature_c": float(self.fabric_temperature_c),
            "heat_pump_electric_kw": electric_kw,
            "delivered_heat_kw": delivered_heat_kw,
            "cop": cop,
            "solar_gain_kw": solar_gain_kw,
        }

