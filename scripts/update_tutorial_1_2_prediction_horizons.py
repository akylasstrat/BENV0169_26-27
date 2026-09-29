"""Add the one-step versus recursive worked example to Tutorial 1.2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATHS = [
    ROOT / "tutorial_1_2_time_series_forecasting.ipynb",
    ROOT / "tutorial_1_2_time_series_forecasting_solution.ipynb",
]
TEMPERATURE_SOURCE_PATH = ROOT / "data" / "tutorial_6_2" / "10mins_solpap_2016.csv"
TEMPERATURE_OUTPUT_PATH = ROOT / "data" / "tutorial_1_2" / "indoor_temperature_10min.csv"


EXPLANATION = r"""### One-step-ahead versus recursive multi-step forecasts

Before comparing day-ahead experiments, consider the simpler persistence rule. The distinction is about **which observations are available when each prediction is made**.

- A **one-step-ahead** forecast predicts one hour, observes the actual outcome, and then resets from that observation before predicting the following hour: $\hat{y}_{t\mid t-1}=y_{t-1}$.
- A **recursive multi-step** forecast is issued once for several future hours. After the first step, the actual outcomes are unavailable, so the method feeds its own previous prediction into the next step: $\hat{y}_{t+h\mid t}=\hat{y}_{t+h-1\mid t}$.

For persistence, the recursive forecast therefore remains equal to the final observation at the forecast origin. The one-step calculation usually appears much more accurate because it receives a new observation every hour. It is useful as a diagnostic, but it is not a fair substitute for a 24- or 168-hour fixed-origin forecast.

The same distinction appears later in the building-model tutorials. A one-step thermal prediction is reset from each measured indoor temperature; a recursive multi-step simulation propagates its own predicted temperature and is called a **free rollout**.
"""


WORKED_EXAMPLE = """# The same persistence rule under two information patterns.
comparison_index = test_week.index

# At each timestamp, the preceding actual load has become available.
one_step_persistence = (
    aggregate_load_2007.shift(1)
    .loc[comparison_index]
    .rename("One-step persistence")
)

# From the fixed origin, no test-week observations are available. Feeding the
# persistence prediction back recursively repeats the final training value.
recursive_persistence = pd.Series(
    training_load.iloc[-1],
    index=comparison_index,
    name="Recursive multi-step persistence",
)

information_comparison = pd.DataFrame(
    {
        "RMSE [MW]": {
            "One-step persistence": np.sqrt(
                np.mean((test_week - one_step_persistence) ** 2)
            ),
            "Recursive multi-step persistence": np.sqrt(
                np.mean((test_week - recursive_persistence) ** 2)
            ),
        },
        "New actual observations used after origin": {
            "One-step persistence": len(test_week) - 1,
            "Recursive multi-step persistence": 0,
        },
    }
)
display(information_comparison.round(1))

comparison_plot_index = comparison_index[:48]
plt.figure(figsize=(11, 4))
plt.plot(
    comparison_plot_index,
    test_week.loc[comparison_plot_index],
    color="black",
    linewidth=1.7,
    label="Actual load",
)
plt.plot(
    comparison_plot_index,
    one_step_persistence.loc[comparison_plot_index],
    label="One-step persistence",
)
plt.plot(
    comparison_plot_index,
    recursive_persistence.loc[comparison_plot_index],
    label="Recursive multi-step persistence",
)
plt.ylabel("Aggregate demand [MW]")
plt.xlabel("Valid time")
plt.title("The same persistence rule with different information availability")
plt.legend()
plt.show()
"""


SES_EXTENSION = r"""## 10. Optional extension — exponential smoothing at 10-minute resolution

High-frequency building measurements are strongly autocorrelated: the next value is often very close to the current value. This can make one-step forecasts appear excellent even when the model provides a poor trajectory over the horizon needed for optimisation.

This extension uses five days of 10-minute average indoor-temperature measurements from an occupied UK house. We train on four days and forecast the final 24 hours using simple exponential smoothing (SES): $\ell_t=\alpha y_t+(1-\alpha)\ell_{t-1}$.

Compare two information patterns using the same fitted method:

- **One-step ahead:** forecast the next 10-minute observation from the current level, observe the actual temperature, and update the level before forecasting again.
- **Recursive multi-step:** freeze the information at the forecast origin and generate all 144 steps without later measurements. For SES without trend or seasonality, feeding forecasts back into the recursion leaves the level unchanged, so the entire trajectory is flat.

The recursive SES forecast does not become numerically unstable; it can nevertheless diverge from the realised temperature as the building warms or cools. Building optimisation normally needs an entire future trajectory from one origin, so one-step accuracy alone is insufficient evidence that a model is suitable for optimisation.

Source: Hollick, F. and Wingfield, J. (2018), *Two periods of in-situ measurements from an occupied, semi-detached house in the UK*. https://doi.org/10.14324/000.ds.10087216
"""


SES_CODE = """TEMPERATURE_PATH = (
    Path("data") / "tutorial_1_2" / "indoor_temperature_10min.csv"
)
temperature_data = pd.read_csv(
    TEMPERATURE_PATH,
    index_col="timestamp",
    parse_dates=True,
).sort_index()

expected_temperature_index = pd.date_range(
    temperature_data.index.min(),
    temperature_data.index.max(),
    freq="10min",
    name="timestamp",
)
if not temperature_data.index.equals(expected_temperature_index):
    raise ValueError("The optional temperature series must have a complete 10-minute index.")
if temperature_data.index.has_duplicates:
    raise ValueError("The optional temperature series contains duplicate timestamps.")

missing_temperature_values = int(
    temperature_data["indoor_temperature_c"].isna().sum()
)
print("Missing indoor-temperature values before interpolation:", missing_temperature_values)

# One isolated value is unavailable. This is a retrospective teaching extract,
# so time interpolation is used before the forecasting comparison.
indoor_temperature = temperature_data["indoor_temperature_c"].interpolate(
    method="time"
)
if indoor_temperature.isna().any():
    raise ValueError("Indoor temperature still contains missing values.")


def ses_one_step_and_recursive(series, forecast_start, alpha):
    # Return one-step-updated and fixed-origin recursive SES forecasts.
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1].")

    training = series.loc[series.index < forecast_start]
    test = series.loc[series.index >= forecast_start]
    if training.empty or test.empty:
        raise ValueError("Both training and test periods must contain observations.")

    level = float(training.iloc[0])
    for observation in training.iloc[1:]:
        level = alpha * float(observation) + (1 - alpha) * level

    # A fixed-origin multi-step SES forecast repeats the final estimated level.
    recursive = pd.Series(level, index=test.index, name="Recursive multi-step SES")

    # One-step evaluation updates the level with each newly observed test value.
    one_step_values = []
    updated_level = level
    for observation in test:
        one_step_values.append(updated_level)
        updated_level = alpha * float(observation) + (1 - alpha) * updated_level
    one_step = pd.Series(
        one_step_values,
        index=test.index,
        name="One-step SES",
    )
    return test, one_step, recursive


ses_alpha = 0.8
ses_forecast_start = indoor_temperature.index.min() + pd.Timedelta(days=4)
ses_actual, ses_one_step, ses_recursive = ses_one_step_and_recursive(
    indoor_temperature,
    forecast_start=ses_forecast_start,
    alpha=ses_alpha,
)

ses_comparison = pd.DataFrame(
    {
        "RMSE [°C]": {
            "One-step SES": np.sqrt(np.mean((ses_actual - ses_one_step) ** 2)),
            "Recursive multi-step SES": np.sqrt(
                np.mean((ses_actual - ses_recursive) ** 2)
            ),
        },
        "Actual observations used after origin": {
            "One-step SES": len(ses_actual) - 1,
            "Recursive multi-step SES": 0,
        },
    }
)
display(ses_comparison.round(3))

fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
axes[0].plot(ses_actual, color="black", linewidth=1.7, label="Measured")
axes[0].plot(ses_one_step, label="One-step SES")
axes[0].plot(ses_recursive, label="Recursive multi-step SES")
axes[0].set_ylabel("Indoor temperature [°C]")
axes[0].set_title(f"Simple exponential smoothing, alpha = {ses_alpha}")
axes[0].legend()

axes[1].plot(
    (ses_actual - ses_one_step).abs(),
    label="Absolute one-step error",
)
axes[1].plot(
    (ses_actual - ses_recursive).abs(),
    label="Absolute recursive error",
)
axes[1].set_ylabel("Absolute error [°C]")
axes[1].set_xlabel("Valid time")
axes[1].legend()
plt.show()
"""


SES_STUDENT_REFLECTION = """### Optional-extension questions

1. Why is the one-step SES error much smaller than the recursive multi-step error?
2. Why is the recursive SES trajectory flat?
3. Which information becomes available to the one-step evaluation but not to the fixed-origin forecast?
4. What additional structure would a temperature model need to represent a changing 24-hour trajectory?
"""


SES_SOLUTION_REFLECTION = """### Optional-extension interpretation

1. The one-step calculation receives a new measured temperature every 10 minutes, so its error has little time to accumulate. The recursive forecast receives no measurements after the origin.
2. SES without trend or seasonality predicts the final estimated level at every future step. Feeding that value back leaves the level unchanged.
3. Each realised test temperature becomes available before the following one-step prediction. None of those test observations is available when the complete 24-hour trajectory is issued.
4. Useful additions could include outdoor temperature, heating inputs, time-of-day effects, an explicit thermal-state model, or trend and seasonal components. Model suitability must be evaluated over the horizon used by the optimiser.
"""


def source_lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source_lines(source),
    }


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source_lines(source),
    }


def build_temperature_extract() -> None:
    raw_data = pd.read_csv(TEMPERATURE_SOURCE_PATH)
    output = raw_data[["Date and time", "T_Average (degC)"]].rename(
        columns={
            "Date and time": "timestamp",
            "T_Average (degC)": "indoor_temperature_c",
        }
    )
    output["timestamp"] = pd.to_datetime(
        output["timestamp"], format="%d/%m/%Y %H:%M"
    )
    output = output.sort_values("timestamp")
    if output["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found in the temperature source.")
    expected_index = pd.date_range(
        output["timestamp"].min(), output["timestamp"].max(), freq="10min"
    )
    if not pd.DatetimeIndex(output["timestamp"]).equals(expected_index):
        raise ValueError("The temperature source does not have a complete 10-minute index.")
    TEMPERATURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(TEMPERATURE_OUTPUT_PATH, index=False, float_format="%.3f")


def update_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    generated_ids = {
        "one-step-recursive-text",
        "one-step-recursive-code",
        "ses-extension-text",
        "ses-extension-code",
        "ses-extension-reflection",
    }
    notebook["cells"] = [
        cell for cell in notebook["cells"] if cell.get("id") not in generated_ids
    ]

    setup_position = next(
        position
        for position, cell in enumerate(notebook["cells"])
        if cell.get("id") == "forecast-setup"
    )
    notebook["cells"][setup_position + 1:setup_position + 1] = [
        markdown_cell(EXPLANATION, "one-step-recursive-text"),
        code_cell(WORKED_EXAMPLE, "one-step-recursive-code"),
    ]

    interpretation_position = next(
        position
        for position, cell in enumerate(notebook["cells"])
        if cell.get("id") == "interpretation-questions"
    )
    solution = path.stem.endswith("_solution")
    notebook["cells"][interpretation_position + 1:interpretation_position + 1] = [
        markdown_cell(SES_EXTENSION, "ses-extension-text"),
        code_cell(SES_CODE, "ses-extension-code"),
        markdown_cell(
            SES_SOLUTION_REFLECTION if solution else SES_STUDENT_REFLECTION,
            "ses-extension-reflection",
        ),
    ]

    # The student notebook is distributed without saved outputs. Existing
    # solution outputs are preserved; the new solution cell is executed later.
    if not path.stem.endswith("_solution"):
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = None
                cell["outputs"] = []

    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    build_temperature_extract()
    for notebook_path in NOTEBOOK_PATHS:
        update_notebook(notebook_path)
    print("Updated Tutorial 1.2 student and solution notebooks.")
