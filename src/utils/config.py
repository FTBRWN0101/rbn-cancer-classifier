from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RBNConfig:
    """Configuration for Random Boolean Network experiments."""
    n_nodes: int
    k: int
    max_steps: int = 100
    random_state: Optional[int] = None

@dataclass(frozen=True)
class NKConfig:
    """Configuration for NK landscape experiments."""
    n: int
    k: int
    sa_steps: int = 1000
    random_state: Optional[int] = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Global experiment configuration."""
    random_state: int
