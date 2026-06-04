from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


def sample_largest_eigenvalue(n: int, p: float) -> float:
    """
    Draw one G(n, p) graph and return its largest adjacency eigenvalue.
    """
    G   = nx.fast_gnp_random_graph(n, p)
    adj = nx.to_scipy_sparse_array(G, dtype=float)

    return float(eigsh(adj, k=1, which="LM", return_eigenvectors=False)[0])


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
