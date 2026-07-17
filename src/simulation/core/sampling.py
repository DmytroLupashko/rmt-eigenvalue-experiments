from __future__ import annotations

import numpy as np
from .ensemble import MatrixEnsemble, BernoulliEnsemble

def sample_largest_eigenvalue(n: int, ensemble: MatrixEnsemble) -> float:
    return float(np.linalg.eigvalsh(ensemble.sample(n))[-1])

def std_error_of(samples: tuple[float, ...]) -> float:
    """Standard error of the mean for the current sample collection."""
    if len(samples) < 2:
        return float("inf")
    return float(np.std(samples) / np.sqrt(len(samples)))


def has_converged(samples: tuple[float, ...], epsilon: float, min_samples: int = 10) -> bool:
    """Return True iff the standard error is below epsilon, and we have enough samples."""
    if len(samples) < min_samples:
        return False
    return std_error_of(samples) < epsilon

if __name__ == '__main__':
    print(sample_largest_eigenvalue(5000, BernoulliEnsemble(0.33)))