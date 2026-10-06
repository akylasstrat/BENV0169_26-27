"""Public constants for BENV0169 Assessment 1, Task 1.

These values are part of the assessment specification. Do not change them.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")

Q2_START = pd.Timestamp("2016-04-01 00:00")
Q2_END = pd.Timestamp("2016-05-05 23:00")

LOAD_TRAIN_START = pd.Timestamp("2016-01-08 00:00")
LOAD_TRAIN_END = pd.Timestamp("2016-09-30 23:00")
LOAD_TEST_START = pd.Timestamp("2016-10-01 00:00")
LOAD_TEST_END = pd.Timestamp("2016-10-28 23:00")

PV_TRAIN_START = pd.Timestamp("2012-04-01 01:00", tz="UTC")
PV_TRAIN_END = pd.Timestamp("2013-03-31 23:00", tz="UTC")
PV_CAL_START = pd.Timestamp("2013-04-01 00:00", tz="UTC")
PV_CAL_END = pd.Timestamp("2013-06-30 23:00", tz="UTC")
PV_TEST_START = pd.Timestamp("2013-07-01 00:00", tz="UTC")
PV_TEST_END = pd.Timestamp("2013-09-30 23:00", tz="UTC")

RF_KWARGS = {
    "n_estimators": 150,
    "min_samples_leaf": 4,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": 1,
}

LOAD_PLOT_START = pd.Timestamp("2016-10-10 00:00")
LOAD_PLOT_END = pd.Timestamp("2016-10-16 23:00")
PV_PLOT_START = pd.Timestamp("2013-08-12 00:00", tz="UTC")
PV_PLOT_END = pd.Timestamp("2013-08-18 23:00", tz="UTC")
PV_INTERVAL_PLOT_START = pd.Timestamp("2013-08-15 00:00", tz="UTC")
PV_INTERVAL_PLOT_END = pd.Timestamp("2013-08-17 23:00", tz="UTC")
