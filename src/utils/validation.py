import numpy as np


def validate_binary_matrix(X: np.ndarray) -> None:
    """Validate that a matrix contains only binary values (0 or 1)."""
    unique_values = np.unique(X)
    if not set(unique_values).issubset({0, 1}):
        raise ValueError("Matrix contains non-binary values: "f"{unique_values}")


def validate_shapes(X: np.ndarray, y: np.ndarray) -> None:
    """Validate feature/label shape compatibility."""
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples, "f"y has {y.shape[0]} labels")
