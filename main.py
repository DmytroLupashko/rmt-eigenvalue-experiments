from pathlib import Path
from typing import Callable
import math

import numpy as np

from src import SimulationConfig, EigenvalueSimulator

# Examples
def log(n: int) -> float:
    return np.log(n)

""" 
def logarithm(base: float = math.e) -> Callable[[float], float]:
    def p_func(n: int) -> float:
        return math.log(n, base)
    p_func.__name__ = f"logarithm"
    return p_func
"""

def power_scaling(beta: float) -> Callable[[int], float]:
    def p_func(n: int) -> float:
        return n ** beta
    p_func.__name__ = f"power_b{beta}"
    return p_func

def linear(a: float, b: float) -> Callable[[int], float]:
    def p_func(n: int) -> float:
        return a * n + b
    p_func.__name__ = f"linear_a={a}, b={b}"
    return p_func



# Entry point
if __name__ == "__main__":
    n_values = range(100, 130, 5)
    results  = []

    for n in n_values:
        config = SimulationConfig(
            n          = n,
            p_func     = power_scaling(0.9), 
            epsilon    = 0.005,
            batch_size = 5000,
            sweep_tag  = "power_0.9_test"
        )

        print(f"Launching simulation  sweep={config.sweep_tag}  n={config.n}")

        result = EigenvalueSimulator(config).run(
            log_to=Path("./data/simulation_results.csv"),
        )

        results.append(result)

    print("Finished simulation!") 