# Tutorial 6.2 occupied-house measurements

The main real-house experiment in Tutorial 6.2 uses two periods of measurements from an occupied semi-detached house in the UK:

- `10mins_solpap_2016.csv`: 11–15 November 2016;
- `10mins_solpap_2017.csv`: 10–14 June 2017.

Each file contains 720 observations at 10-minute resolution. The experiment uses average indoor temperature, outdoor temperature, total power and solar irradiance. The notebook fills missing readings by linear interpolation, separately within each fitting and test period. The original CSV files remain unchanged.

`P_tot (W)` combines measured electricity and gas power and serves as a proxy for heat entering the building. `Solar (W/m2)` measures irradiance. Both are divided by 1000 for the regression, giving kW and kW/m². The fitted effective solar-gain factor has units of m², so its product with irradiance is a heat gain in kW. The recovered resistance, capacitance and solar-gain factor describe this simplified model under the heat-proxy assumption.

The November model uses all five November days. The June model uses 10–12 June; both models are evaluated on the same transitions within 13–14 June. Splitting observations before interpolation and target construction keeps the test period separate. The counts are 719 November fitting transitions, 431 June fitting transitions and 287 June test targets. The teaching focus is the effect of model simplification and training data on effective coefficients and prediction performance.

Source: Hollick, F. and Wingfield, J. (2018), *Two periods of in-situ measurements from an occupied, semi-detached house in the UK*. https://doi.org/10.14324/000.ds.10087216
