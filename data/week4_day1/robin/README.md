# Tutorial 4.1a: Robin / UCL alternative

This alternative uses **Robin_education_Margarito** in **2017**, a college laboratory at the Robin site (University College London). The pseudonym is the dataset identifier, not the building's public name. Metadata report a floor area of 8,468 m² and the Europe/London timezone.

## Sources and references

Miller, C., Kathirgamanathan, A., Picchetti, B. et al. (2020), [‘The Building Data Genome Project 2, energy meter data from the ASHRAE Great Energy Predictor III competition’](https://doi.org/10.1038/s41597-020-00712-x), *Scientific Data*, 7, 368.

- [BDG2 dataset and documentation](https://github.com/buds-lab/building-data-genome-project-2)
- [Archived dataset release](https://zenodo.org/records/3887306)
- [Meter units and source cleaning](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features)

The extract retains the CC BY-SA 4.0 source licence and attribution. It was prepared from the supplied local BDG2 checkout, without downloading new data or modifying that repository. `provenance.json` records source paths, checksums, extraction rules and versions.

## Selected building

`robin_education_margarito_2017.csv` contains 8,760 hourly rows, 1 January–31 December 2017. Columns:

| Column | Meaning |
|---|---|
| `timestamp` | Recorded building timestamp; supplied hourly index retained |
| `electricity_kwh` | Electricity energy per hourly interval, kWh |
| `hotwater_kwh` | Thermal energy delivered through hot water per hourly interval, kWh; not water volume |
| `air_temperature_c` | Observed Robin-site air temperature, °C |

The energy columns come from the **cleaned**, full-precision electricity and hot-water tables, not the Kaggle or memory-efficient tables. Source cleaning has already removed selected anomalous readings; the tutorial does not infer their cause. Weather is measured, not forecast. The extract has no missing electricity readings, 24 missing hot-water readings and seven missing temperatures. Source values, including heat-meter zeros, are retained. There are no duplicate or absent hourly timestamps. No timezone conversion or imputation is applied to this extract.

Training targets are **8 January–30 September 2017 (6,384 rows)**, preceded by seven days of lag history. Test targets are **1–28 October 2017 (672 rows)**. Issue forecasts at 23:00 for all 24 hours of the following day. Training and test electricity are positive and complete; all lag-24/168 inputs are available. No load reconstruction or target exclusion is needed. One hot-water reading and seven aligned temperature readings are missing in training; both descriptive series are complete in the test period. Neither is a forecasting input. The temperature scatter uses 6,377 complete observed pairs, while forecasting retains all 6,384 training targets. The natural heat-meter gap is shown without changing electricity targets. Gas is not provided.

The initial meter plot and resolution example use **6–12 February 2017**, a complete training week with variation in both meters. Six example rows come from 6 February, 08:00–13:00. Forecast plots show 8–14 October 2017; scores use all 28 test days.

## Why the building and year changed

NaN completeness alone was insufficient for the earlier Cecilia example. Its 2016 source contains an exact 1,043-hour constant stretch. The same values are present in both BDG2 raw and cleaned files; this tutorial did not fill them. They may reflect upstream imputation or another issue, but their cause is unverified. Cecilia also has a 120-hour constant stretch in 2017.

`building_quality_comparison.csv` records NaNs, nonpositive test readings and identical-value runs for the Robin buildings with hot-water meters in 2016/2017. Margarito's full 2017 electricity has no NaNs and a longest identical-value run of two hours. We selected it using these source-quality checks, before comparing forecasts. Passing these checks does not certify that every value is untouched upstream. The previous `robin_education_cecilia.csv` is retained as an unused historical extract; current notebooks do not load it.

## Profiles by building usage

`robin_building_metadata.csv` supplies selected metadata for all 52 Robin buildings. `robin_usage_profiles.csv` contains 2,304 rows: 48 retained buildings × two day types × 24 hours. It covers 10 lodging/residential, 22 education and 16 office buildings. Assembly/library buildings are outside the three-category comparison. Robin_office_Erma has no usable training readings after screening and is excluded rather than assigned a fabricated profile.

Only readings from **8 January–30 September 2017** are used. Within each building's training series, flag exact constant runs lasting at least 24 hours. Exclude these values and source NaNs from descriptive means; do not alter the source or interpolate across them. This is a conservative quality rule, not proof that every flagged reading was imputed. Run identification uses training only, including at its boundaries. Across the 49 candidate buildings there are 6,647 source NaNs and 8,014 flagged non-missing hours. `robin_profile_quality.csv` reports exclusions and retained counts per building.

Calculate each retained building's mean at each hour separately for weekdays and weekends. `n_observed` records the retained sample size. `training_max_kwh` is that building's maximum retained training reading. `normalised_mean = mean_electricity_kwh / training_max_kwh`. No profile values are filled.

Grey plot lines represent individual building mean profiles. Black lines are the equal-weight mean of these profiles within each usage category; larger buildings do not receive more weight. This normalisation compares relative shapes, not absolute consumption or energy efficiency. All peaks used for scaling come from training, not test. The compact table avoids distributing all 49 hourly series; the instructor utility can reconstruct it from the source files.

## Run and regenerate

The revised lecture uses this Robin version as its main building example. Eight short exercises cover daily totals, weekday/weekend means, the load-duration curve, two demand indicators, training-only scaling and the three learnt model fits. The weekly seasonal benchmark, feature construction and plotting remain supplied. Ridge uses cyclical hour/weekday inputs; both tree-based methods use identical integer calendar inputs. The encoding explanation appears here only, not in the PV notebook. The earlier hourly-boxplot exercise is not part of this version.

## Selected building group

`robin_group_electricity_2017_week.csv` contains original, aligned hourly electricity readings for Robin_education_Margarito (laboratory), Robin_lodging_Armand (residence) and Robin_office_Sammie (office), over 6–12 February 2017. It has 168 complete hourly rows and three electricity columns, all in kWh per one-hour interval. Timestamps retain the supplied Robin clock.

Margarito is the running example. Armand and Sammie are the alphabetically first buildings in their respective usage categories with complete 8 January–30 September training readings and no exact constant run of 24 hours or more. The group and week were fixed from metadata and data quality before examining aggregate peaks or forecast results. This is a selected building group, not a documented shared feeder.

No readings are imputed, normalised or replaced with zero. Sum columns at the same timestamp. Divide the one-hour interval energy by one hour to obtain hourly average kW. The peak of this simultaneous sum and the sum of separate peaks use identical timestamps. Aggregation changes peak coincidence, not the group's total energy. Source identity, selection and extract checksum are recorded in `provenance.json`.

## Execution

Launch Jupyter from the teaching repository root. Neither notebook imports external helper/configuration files. The alternative has its own figures in `figures/week4_day1/robin/`, manifest and restart files, so running it does not replace the original tutorial's outputs.

Instructor preparation, if source extracts need to be rebuilt:

```bash
python teaching/build_robin_alternative.py /path/to/building-data-genome-project-2 --data-only
python -m jupyter nbconvert --to notebook --execute --inplace tutorial_4_1a_building_load_robin_solution.ipynb --ExecutePreprocessor.timeout=600
```

Use `--data-only` to rebuild source extracts without changing the edited notebooks. The legacy notebook-generation mode is disabled by default because it predates the revised lecture. Execute the Robin and PV solutions, then run `python teaching/validate_week4_day1.py` to regenerate and validate the current lecture manifest.
