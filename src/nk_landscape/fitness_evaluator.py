import numpy as np
from nk_landscape import NKLandscape

class FitnessEvaluator:
    """Wrapper for NK landscape fitness evaluation."""
    def __init__(self, landscape: NKLandscape):
        self.landscape = landscape
    def evaluate(self, genome: np.ndarray) -> float:
        return self.landscape.fitness(genome)
