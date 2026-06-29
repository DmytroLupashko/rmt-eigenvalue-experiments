from __future__ import annotations

from line_profiler import profile
import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

DENSE_THRESHOLD = 2000  # tune empirically for your hardware

@profile
def sample_largest_eigenvalue(n: int, p: float) -> float:
    if n < DENSE_THRESHOLD:
        # Dense NumPy path: fast generation + LAPACK
        upper = np.random.random((n, n)) < p
        adj = np.triu(upper, k=1)
        adj = (adj + adj.T).astype(float)
        return float(np.linalg.eigvalsh(adj)[-1])
    else:
        # Sparse path: memory-efficient generation + ARPACK
        G = nx.fast_gnp_random_graph(n, p)
        adj = nx.to_scipy_sparse_array(G, dtype=float, format="csr")

        v0 = np.ones(n) / np.sqrt(n)  # warm start: near-optimal for Erdős–Rényi
        return float(eigsh(adj, k=1, which="LM",
                           return_eigenvectors=False,
                           v0=v0, tol=1e-4)[0])


# def sample_largest_eigenvalue(n: int, p: float) -> float:
#     """
#     Draw one G(n, p) graph and return its largest adjacency eigenvalue.
#     """
#     G   = nx.fast_gnp_random_graph(n, p)
#     adj = nx.to_scipy_sparse_array(G, dtype=float)
#
#     return float(eigsh(adj, k=1, which="LM", return_eigenvectors=False)[0])


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
    print(sample_largest_eigenvalue(5000, 0.33))