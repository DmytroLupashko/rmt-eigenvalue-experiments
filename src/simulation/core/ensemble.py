from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


class MatrixEnsemble(ABC):
    """Generates symmetric random matrices for a given size n."""

    @abstractmethod
    def sample(self, n: int) -> np.ndarray:
        """Return a symmetric n×n random matrix."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in sweep_tag and logging."""
        ...


@dataclass(frozen=True)
class BernoulliEnsemble(MatrixEnsemble):
    """Unnormalized Erdős–Rényi adjacency matrix"""

    p_func_name: str
    p: float

    def sample(self, n: int) -> np.ndarray:
        upper = np.random.binomial(1, self.p, size=(n, n)).astype(float)
        matrix = np.triu(upper, k=1)
        return matrix + matrix.T

    @property
    def name(self) -> str:
        return f"bernoulli_{self.p_func_name}"


@dataclass(frozen=True)
class NormalizedBernoulliEnsemble(MatrixEnsemble):
    """Bernoulli adjacency matrix normalized by 1/sqrt(n·p·(1-p))."""

    p_func_name: str
    p: float

    def sample(self, n: int) -> np.ndarray:
        upper = np.random.binomial(1, self.p, size=(n, n)).astype(float)
        matrix = np.triu(upper, k=1)
        matrix = matrix + matrix.T

        # Normalization
        matrix -= self.p * (np.ones((n, n)) - np.eye(n))
        sigma = np.sqrt(n * self.p)

        return matrix / sigma

    @property
    def name(self) -> str:
        return f"bernoulli_normalized_{self.p_func_name}"


@dataclass(frozen=True)
class GaussianEnsemble(MatrixEnsemble):
    """GOE-style Wigner matrix: off-diagonal ~ N(0,1), diagonal ~ N(0,2), normalized by 1/sqrt(n)."""

    def sample(self, n: int) -> np.ndarray:
        upper = np.random.normal(0.0, 1.0, size=(n, n))
        matrix = np.triu(upper, k=1)
        matrix = matrix + matrix.T
        return matrix / np.sqrt(n)

    @property
    def name(self) -> str:
        return "gaussian"


@dataclass(frozen=True)
class UniformEnsemble(MatrixEnsemble):
    """Wigner matrix with entries ~ Uniform(-1, 1), normalized by 1/sqrt(n)."""

    def sample(self, n: int) -> np.ndarray:
        upper = np.random.uniform(-1.0, 1.0, size=(n, n))
        matrix = np.triu(upper, k=1)
        matrix = matrix + matrix.T
        np.fill_diagonal(matrix, np.random.uniform(-1.0, 1.0, size=n))
        return matrix / np.sqrt(n)

    @property
    def name(self) -> str:
        return "uniform"