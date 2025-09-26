"""Simple MLflow-like stub for logging parameters and metrics.

This module provides a minimalistic replacement for MLflow's tracking API.
It allows you to log parameters and metrics for different model runs and
saves them to JSON files under the `mlruns` directory. Each experiment
contains subdirectories for individual runs. This stub is intended to
approximate the functionality of MLflow in environments where the
official package is unavailable.

Usage:

```
import mlflow_stub as mlflow

mlflow.set_experiment("my_experiment")
with mlflow.start_run(run_name="run1") as run:
    mlflow.log_param("C", 1.0)
    mlflow.log_metric("accuracy", 0.85)
```
```
"""

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

_current_experiment = "default"

def set_experiment(name: str) -> None:
    """Set the current experiment name.

    Args:
        name: Name of the experiment. A directory with this name will be
              created under `mlruns/` if it does not already exist.
    """
    global _current_experiment
    _current_experiment = name
    os.makedirs(os.path.join("mlruns", name), exist_ok=True)


class _Run:
    """Represents a single run for tracking parameters and metrics."""

    def __init__(self, experiment: str, run_id: str, run_name: Optional[str] = None):
        self.experiment = experiment
        self.run_id = run_id
        self.run_name = run_name or run_id
        self.run_dir = os.path.join("mlruns", experiment, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.utcnow().isoformat()

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        self.params[key] = value

    def log_metric(self, key: str, value: Any) -> None:
        """Log a single metric.

        Args:
            key: Metric name.
            value: Metric value.
        """
        self.metrics[key] = value

    def _write_run_info(self) -> None:
        """Persist parameters and metrics to JSON files."""
        info = {
            "run_name": self.run_name,
            "experiment": self.experiment,
            "run_id": self.run_id,
            "start_time": self.start_time,
            "params": self.params,
            "metrics": self.metrics,
        }
        with open(os.path.join(self.run_dir, "run_info.json"), "w") as f:
            json.dump(info, f, indent=2)

    def end_run(self) -> None:
        """Finalize the run and write information to disk."""
        self._write_run_info()

    # Provide context manager support
    def __enter__(self) -> "_Run":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.end_run()


def start_run(run_name: Optional[str] = None) -> _Run:
    """Start a new run for the current experiment.

    Args:
        run_name: Optional human-friendly name for the run.

    Returns:
        An instance of `_Run` used to log parameters and metrics.
    """
    run_id = str(uuid.uuid4())[:8]
    return _Run(_current_experiment, run_id, run_name)


def log_param(key: str, value: Any) -> None:
    """Helper to log a parameter on the active run.

    This function is provided for compatibility with MLflow's API, but in
    this simple stub, users should prefer using the `log_param` method of
    the returned run object directly.
    """
    raise RuntimeError(
        "Global log_param is not supported. Use the `log_param` method of the run object."
    )


def log_metric(key: str, value: Any) -> None:
    """Helper to log a metric on the active run.

    Raises because global logging is not supported.
    """
    raise RuntimeError(
        "Global log_metric is not supported. Use the `log_metric` method of the run object."
    )