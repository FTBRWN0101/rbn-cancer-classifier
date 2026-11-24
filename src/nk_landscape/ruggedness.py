import numpy as np

def autocorrelation(fitness_values: np.ndarray) -> float:
    """Compute autocorrelation as a ruggedness metric."""
    if len(fitness_values) < 2:
        return 0.0

    return float(np.corrcoef(fitness_values[:-1], fitness_values[1:])[0, 1])