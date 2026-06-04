from pathlib import Path
from typing import Callable

import numpy as np

from src import SimulationConfig, EigenvalueSimulator

# Examples
def log(n: int) -> float:
    return np.log(n)

def power_scaling(beta: float) -> Callable[[int], float]:
    def p_func(n: int) -> float:
        return n ** beta
    p_func.__name__ = f"power_b{beta}"
    return p_func


# Entry point
if __name__ == "__main__":
    config = SimulationConfig(
        n          = 100,
        p_func     = power_scaling(1/2),
        epsilon    = 0.005,
        batch_size = 10,
    )

    result = EigenvalueSimulator(config).run(
        log_to=Path("./data/simulation_results.csv"),
    )

    print(
        f"\nFinal result : {result.mean_eigenvalue:.4f}"
        f"  ±{result.std_error:.4f}"
        f"  ({result.samples} samples, {result.elapsed_seconds:.1f}s)"
    )