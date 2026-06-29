from line_profiler import profile
import time
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor

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
        self._executor = ProcessPoolExecutor()

    @staticmethod
    def sample_eigenvalues_batch(n, p, count) -> list[float]:
        """One worker computes `count` eigenvalues, reducing IPC calls."""
        return [sample_largest_eigenvalue(n, p) for _ in range(count)]

    @profile
    def _collect_batch(self) -> list[float]:
        """
        Draw batch_size independent eigenvalue samples in parallel using
        ProcessPoolExecutor. Each worker process runs sample_largest_eigenvalue
        independently, bypassing the GIL for true CPU-level parallelism.
        """
        futures = [
            self._executor.submit(sample_largest_eigenvalue, self.config.n, self.config.p)
            for _ in range(self.config.batch_size)
        ]
        return [f.result() for f in futures]

    def run(self, log_to: Path | None = LOG_PATH) -> SimulationResult:
        """
        Run batches until convergence, then optionally log and return a SimulationResult.

        Parameters
        ----------
        log_to:
            Path to the CSV file. Pass None to skip logging.
        """
        cfg = self.config

        start = time.perf_counter()
        samples: tuple[float, ...] = ()

        while not has_converged(samples, cfg.epsilon) or len(samples) < cfg.min_samples:
            samples = samples + tuple(self._collect_batch())

        elapsed = time.perf_counter() - start

        result = SimulationResult(
            config=cfg,
            mean_eigenvalue=float(np.mean(samples)),
            std_error=std_error_of(samples),
            samples=samples,
            samples_size=len(samples),
            elapsed_seconds=elapsed,
        )

        if log_to is not None:
            log_result(result, path=log_to)

        return result
