"""Build the instructor-only BOPTEST REST API smoke-test notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "instructor" / "boptest_smoke_test.ipynb"


def markdown(source):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source):
    return nbf.v4.new_code_cell(source.strip())


def build_notebook():
    cells = [
        markdown(
            """
# Instructor smoke test: BOPTEST

This notebook checks a local BOPTEST deployment through its REST API. It is deliberately separate from the student tutorials: BOPTEST remains a future reference-plant option, while the taught notebooks run offline with the supplied two-state plant.

The test selects `bestest_hydronic_heat_pump`, checks its points, requests a weather forecast, advances the emulator under its default controller and then applies a short proportional override. It does not implement MPC or reinforcement learning.

The notebook follows the BOPTEST v0.9.0 API. Run it only when a local BOPTEST service is available.
"""
        ),
        markdown(
            """
## 1. Start the local service

From the root of a current `project1-boptest` checkout, run:

```bash
docker compose up web worker provision
```

The default local API is `http://127.0.0.1:80`. Set the environment variable `BOPTEST_URL` if the service uses another host or port. When testing finishes, run `docker compose down` from the BOPTEST repository.

This notebook registers a best-effort cleanup function, but always run the final stop cell as well.
"""
        ),
        code(
            """
import atexit
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_URL = os.environ.get("BOPTEST_URL", "http://127.0.0.1:80").rstrip("/")
TEST_CASE = "bestest_hydronic_heat_pump"
TIMEOUT_SECONDS = 90

print("BOPTEST URL:", BASE_URL)
print("Test case:", TEST_CASE)
"""
        ),
        markdown("## 2. Small REST client"),
        code(
            """
def api_request(method, endpoint, *, payload=None, timeout=TIMEOUT_SECONDS):
    '''Send one BOPTEST request and return its payload.'''
    encoded_payload = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        f"{BASE_URL}/{endpoint.lstrip('/')}",
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urlopen(request, timeout=timeout) as response:
        response_bytes = response.read()
        if not response_bytes:
            return None
        body = json.loads(response_bytes.decode("utf-8"))
    if isinstance(body, dict) and "status" in body:
        if body["status"] != 200:
            raise RuntimeError(f"BOPTEST error {body['status']}: {body.get('message', '')}")
        return body.get("payload")
    return body


test_id = None


def stop_test():
    '''Free the BOPTEST worker if this notebook selected a test case.'''
    global test_id
    if test_id is None:
        return
    try:
        api_request("PUT", f"stop/{test_id}", timeout=30)
        print("Stopped test case:", test_id)
    except Exception as error:
        print("Could not stop the test case automatically:", error)
    finally:
        test_id = None


atexit.register(stop_test)
"""
        ),
        markdown("## 3. Connect and select the test case"),
        code(
            """
try:
    selection = api_request("POST", f"testcases/{TEST_CASE}/select")
except (URLError, TimeoutError) as error:
    raise ConnectionError(
        f"No BOPTEST service responded at {BASE_URL}. Start the local Docker service "
        "or set BOPTEST_URL, then run this cell again."
    ) from error

test_id = selection["testid"] if isinstance(selection, dict) else selection
version = api_request("GET", f"version/{test_id}")
print("BOPTEST version:", version)
print("Allocated test ID:", test_id)
"""
        ),
        markdown("## 4. Inspect the interface before sending controls"),
        code(
            """
measurements = api_request("GET", f"measurements/{test_id}")
inputs = api_request("GET", f"inputs/{test_id}")
forecast_points = api_request("GET", f"forecast_points/{test_id}")

required_measurements = {"reaTZon_y"}
required_inputs = {"oveHeaPumY_u", "oveHeaPumY_activate"}
missing_measurements = required_measurements.difference(measurements)
missing_inputs = required_inputs.difference(inputs)
if missing_measurements or missing_inputs:
    raise KeyError(
        "The test-case interface differs from this smoke test. "
        f"Missing measurements={sorted(missing_measurements)}, "
        f"missing inputs={sorted(missing_inputs)}."
    )

print(f"Measurements: {len(measurements)}")
print(f"Inputs: {len(inputs)}")
print(f"Forecast points: {len(forecast_points)}")
display(pd.DataFrame(inputs).T.loc[sorted(required_inputs)])
"""
        ),
        markdown(
            """
## 5. Initialise a short, repeatable experiment

The test case provides `typical_heat_day` and `peak_heat_day`. This smoke test uses the typical heating day, dynamic electricity prices and a one-hour control step.
"""
        ),
        code(
            """
initial_measurement = api_request(
    "PUT",
    f"scenario/{test_id}",
    payload={"time_period": "typical_heat_day", "electricity_price": "dynamic"},
)
step_seconds = api_request("PUT", f"step/{test_id}", payload={"step": 3600})

print("Control step [s]:", step_seconds)
print(f"Initial zone temperature: {initial_measurement['reaTZon_y'] - 273.15:.2f} °C")
"""
        ),
        markdown("## 6. Request one boundary-condition forecast"),
        code(
            """
preferred_forecast_points = [
    "TDryBul",
    "HGloHor",
    "PriceElectricPowerDynamic",
    "LowerSetp[1]",
    "UpperSetp[1]",
]
selected_forecast_points = [
    point for point in preferred_forecast_points if point in forecast_points
]
if not selected_forecast_points:
    selected_forecast_points = list(forecast_points)[:3]

forecast = api_request(
    "PUT",
    f"forecast/{test_id}",
    payload={
        "point_names": selected_forecast_points,
        "horizon": 24 * 3600,
        "interval": 3600,
    },
)
forecast_frame = pd.DataFrame(forecast)
assert len(forecast_frame) >= 24
display(forecast_frame.head())
"""
        ),
        markdown(
            """
## 7. Advance with the embedded controller

An empty input dictionary leaves the test case's embedded controller active. This first loop checks that the emulator advances and returns finite measurements.
"""
        ),
        code(
            """
default_records = []
for _ in range(6):
    measurement = api_request("POST", f"advance/{test_id}", payload={})
    default_records.append(measurement)

default_results = pd.DataFrame(default_records)
assert np.isfinite(default_results["reaTZon_y"]).all()
display(default_results[["time", "reaTZon_y"]].assign(
    zone_temperature_c=lambda frame: frame["reaTZon_y"] - 273.15
))
"""
        ),
        markdown(
            r"""
## 8. Apply a short proportional heat-pump override

The controller below is intentionally simple. It maps the zone-temperature error to the normalised heat-pump modulation $u$ and clips the command to $0 \leq u \leq 1$.
"""
        ),
        code(
            """
setpoint_c = 21.0
gain_per_c = 0.6
measurement = default_records[-1]
controlled_records = []

for _ in range(24):
    zone_temperature_c = measurement["reaTZon_y"] - 273.15
    command = float(np.clip(gain_per_c * (setpoint_c - zone_temperature_c), 0.0, 1.0))
    measurement = api_request(
        "POST",
        f"advance/{test_id}",
        payload={"oveHeaPumY_u": command, "oveHeaPumY_activate": 1.0},
    )
    controlled_records.append({**measurement, "command": command})

controlled_results = pd.DataFrame(controlled_records)
assert controlled_results["command"].between(0, 1).all()
assert np.isfinite(controlled_results["reaTZon_y"]).all()

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
axes[0].plot(controlled_results["time"] / 3600, controlled_results["reaTZon_y"] - 273.15)
axes[0].axhline(setpoint_c, color="black", linestyle="--", label="Setpoint")
axes[0].set_ylabel("Zone temperature [°C]")
axes[0].legend()
axes[1].step(controlled_results["time"] / 3600, controlled_results["command"], where="post")
axes[1].set_ylabel("Heat-pump command [-]")
axes[1].set_xlabel("Simulation time [h]")
plt.show()
"""
        ),
        markdown("## 9. Inspect BOPTEST KPIs and stop the test"),
        code(
            """
kpis = api_request("GET", f"kpi/{test_id}")
display(pd.Series(kpis, name="value").to_frame())

stop_test()
"""
        ),
        markdown(
            """
## Pass criteria

The smoke test passes when the notebook:

- reports a BOPTEST version and allocates a test ID;
- finds the expected heat-pump input and zone-temperature output;
- returns a 24-hour forecast;
- advances under both the embedded and proportional controllers with finite values;
- reports KPIs; and
- stops the selected test case.

If a future BOPTEST release changes signal names or API paths, update this notebook before using BOPTEST as the reference plant for the student tutorials.
"""
        ),
    ]

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    }
    return notebook


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUTPUT)


if __name__ == "__main__":
    main()
