from __future__ import annotations

import datetime
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .core.ensemble import MatrixEnsemble

PFunc = Callable[[int], float]


@dataclass(frozen=True)
class SimulationConfig:
    """All parameters that define a single simulation run. Immutable by design."""

    n: int
    ensemble: MatrixEnsemble
    epsilon: float
    min_samples: int
    batch_size: int = 1000
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sweep_tag: str = ""

@dataclass(frozen=True)
class SimulationResult:
    """Final outcome of one converged simulation run."""

    config: SimulationConfig
    mean_eigenvalue: float
    std_error: float
    samples: tuple[float, ...]
    samples_size: int
    elapsed_seconds: float
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
