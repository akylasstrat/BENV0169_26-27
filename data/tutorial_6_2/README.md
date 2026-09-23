# Tutorial 6.2 occupied-house measurements

The optional Tutorial 6.2 extension uses two periods of measurements from an occupied semi-detached house in the UK:

- `10mins_solpap_2016.csv`: 11–15 November 2016;
- `10mins_solpap_2017.csv`: 10–14 June 2017.

Each file contains 720 observations at 10-minute resolution. The extension uses average indoor temperature, outdoor temperature, total power and solar irradiance. Short gaps are retained in these source files and handled explicitly in the notebook.

`P_tot (W)` combines measured electricity and gas power. It is used as a proxy for heat entering the building and is not a direct measurement of useful space-heating output. Parameters identified from this proxy should therefore be interpreted as effective model parameters.

Source: Hollick, F. and Wingfield, J. (2018), *Two periods of in-situ measurements from an occupied, semi-detached house in the UK*. https://doi.org/10.14324/000.ds.10087216

These copies were taken from the BENV0092 teaching repository. The BENV0092 repository itself has not been modified.
