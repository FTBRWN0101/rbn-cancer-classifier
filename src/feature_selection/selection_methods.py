import numpy as np
from .importance import variance_importance

def top_k_variance(X: np.ndarray, k: int) -> np.ndarray:
    scores = variance_importance(X)
    return np.argsort(scores)[::-1][:k]
