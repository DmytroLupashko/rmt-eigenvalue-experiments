import time
from pathlib import Path

import numpy as np

from .core.sampling import sample_largest_eigenvalue, std_error_of, has_converged
from .io.logging import LOG_PATH, log_result
from .config import SimulationConfig, SimulationResult


class EigenvalueSimulator:
    """
    Monte Carlo estimator of the spectral radius of a random Erdős–Rényi graph.

    Runs batches of samples sequentially until the standard error of the mean
    falls below the configured epsilon threshold.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

    def _collect_batch(self) -> list[float]:
        """Draw batch_size independent eigenvalue samples and return them as a list."""
        return [
            sample_largest_eigenvalue(self.config.n, self.config.p)
            for _ in range(self.config.batch_size)
        ]

    def run(self, log_to: Path | None = LOG_PATH) -> SimulationResult:
        """
        Run batches until convergence, then optionally log and return a SimulationResult.

        Parameters
        ----------
        log_to:
            Path to the CSV file. Pass None to skip logging.
        """
        cfg = self.config
        print(f"Launching simulation  n={cfg.n}  p={cfg.p:.5f}  ε={cfg.epsilon}")

        start   = time.perf_counter()
        samples: tuple[float, ...] = ()

        while not has_converged(samples, cfg.epsilon):
            samples = samples + tuple(self._collect_batch())
            print(f"  samples={len(samples):4d}  std_error={std_error_of(samples):.6f}")

        elapsed = time.perf_counter() - start
        print(f"Converged after {len(samples)} samples ({elapsed:.1f}s)")

        result = SimulationResult(
            config          = cfg,
            mean_eigenvalue = float(np.mean(samples)),
            std_error       = std_error_of(samples),
            samples         = len(samples),
            elapsed_seconds = elapsed,
        )

        if log_to is not None:
            log_result(result, path=log_to)
            print(f"Result logged → {log_to.resolve()}")

        return result