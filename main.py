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
    n_values = range(100, 200, 5)
    results  = []

    for n in n_values:
        config = SimulationConfig(
            n          = n,
            p_func     = power_scaling(1/2),
            epsilon    = 0.005,
            batch_size = 5000,
        )

        print(f"Launching simulation p_func={config.p_func.__name__}  n={config.n}  p={config.p:.5f}  ε={config.epsilon}")

        result = EigenvalueSimulator(config).run(
            log_to=Path("./data/simulation_results.csv"),
        )

        results.append(result)

    print("Finished simulation!")