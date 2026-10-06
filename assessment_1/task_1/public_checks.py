"""Small feedback checks for BENV0169 Assessment 1, Task 1.

The checks use synthetic inputs and do not reveal answers for the released data.
Passing them is useful feedback, not a mark or proof that the whole task is correct.
"""

import numpy as np
import pandas as pd


def check_metric_functions(mae_function, rmse_function):
    index = pd.date_range("2020-01-01", periods=3, freq="h")
    actual = pd.Series([1.0, 2.0, 5.0], index=index)
    forecast = pd.Series([2.0, 2.0, 2.0], index=index)
    assert np.isclose(mae_function(actual, forecast), 4 / 3)
    assert np.isclose(rmse_function(actual, forecast), np.sqrt(10 / 3))
    try:
        rmse_function(actual, forecast.set_axis(index + pd.Timedelta(hours=1)))
    except ValueError:
        pass
    else:
        raise AssertionError("RMSE must reject misaligned pandas indices")
    print("Metric feedback checks passed.")


def check_imputation_function(function, expected):
    index = pd.date_range("2020-01-01", periods=5, freq="h")
    source = pd.Series([1.0, np.nan, np.nan, 4.0, 5.0], index=index)
    result = function(source.copy())
    assert result.index.equals(source.index)
    assert result[source.notna()].equals(source[source.notna()])
    assert np.allclose(result.to_numpy(), np.asarray(expected), equal_nan=False)


def check_unit_interval_projection(function):
    values = pd.Series([-0.2, 0.0, 0.4, 1.0, 1.3])
    expected = pd.Series([0.0, 0.0, 0.4, 1.0, 1.0])
    result = function(values)
    assert np.allclose(result, expected)
    print("PV-bound feedback check passed.")


def check_interval_metrics(coverage_function, winkler_function):
    actual = pd.Series([0.0, 0.5, 1.0])
    lower = pd.Series([0.0, 0.2, 0.6])
    upper = pd.Series([0.1, 0.8, 0.9])
    assert np.isclose(coverage_function(actual, lower, upper), 200 / 3)
    expected_winkler = np.mean([0.1, 0.6, 0.3 + 20 * 0.1])
    assert np.isclose(winkler_function(actual, lower, upper), expected_winkler)
    print("Interval-metric feedback checks passed.")
