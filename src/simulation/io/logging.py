from __future__ import annotations

import csv
from pathlib import Path

from ..config import SimulationResult

LOG_PATH = Path(__file__).parent.parent.parent / "data" / "simulation_results.csv"

_CSV_HEADER = [
    "p_func",
    "n",
    "mean_eigenvalue",
    "epsilon",
    "_batch_size",
    "_elapsed_time",
    "_timestamp",
    "_std_error"
]

def _ensure_log_header(path: Path) -> None:
    """Create the CSV file with a header row if it does not yet exist."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        with path.open("w", newline="") as fh:
            csv.writer(fh).writerow(_CSV_HEADER)


def log_result(result: SimulationResult, path: Path = LOG_PATH) -> None:
    """
    Append one result row to the CSV log.

    Columns: p_func, n, mean_eigenvalue, epsilon,
             _batch_size, _elapsed_time, _timestamp, _std_error.
    """
    _ensure_log_header(path)
    cfg = result.config
    row = [
        cfg.p_func.__name__,
        cfg.n,
        f"{result.mean_eigenvalue:.6f}",
        cfg.epsilon,
        result.samples,
        f"{result.elapsed_seconds:.2f}",
        result.timestamp.isoformat(timespec="seconds"),
        f"{result.std_error:.6f}",
    ]
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow(row)
