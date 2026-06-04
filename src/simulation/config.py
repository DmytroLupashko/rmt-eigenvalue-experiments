from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable

PFunc = Callable[[int], float]

@dataclass(frozen=True)
class SimulationConfig:
    """All parameters that define a single simulation run. Immutable by design."""

    n:          int
    p_func:     PFunc
    epsilon:    float
    batch_size: int  = 10

    @property
    def p(self) -> float:
        """Edge probability derived from the scaling function."""
        return self.p_func(self.n) / self.n


@dataclass(frozen=True)
class SimulationResult:
    """Final outcome of one converged simulation run."""

    config:          SimulationConfig
    mean_eigenvalue: float
    std_error:       float
    samples:         int
    elapsed_seconds: float
    timestamp:       datetime.datetime = field(default_factory=datetime.datetime.now)

