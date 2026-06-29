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

def p_n(n: int) -> float:
    return n


# Entry point
if __name__ == "__main__":
    config = SimulationConfig(
        n=300,
        p_func=power_scaling(0.5),
        epsilon=0.05,
        min_samples=0,
        batch_size=5000,
        sweep_tag="lol_kek"
    )

    print(f"Launching simulation  sweep={config.sweep_tag}  n={config.n}")

    result = EigenvalueSimulator(config).run(
        log_to=Path("./data/simulation_results.csv"),
    )