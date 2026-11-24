import numpy as np
from .fitness_evaluator import FitnessEvaluator


class SimulatedAnnealing:
    """Simulated Annealing optimiser for NK landscapes."""

    def __init__(
        self,
        evaluator: FitnessEvaluator,
        initial_temperature: float = 1.0,
        cooling_rate: float = 0.995,
        steps: int = 1000):
        self.evaluator = evaluator
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.steps = steps

    def optimise(self, genome: np.ndarray) -> np.ndarray:
        current = genome.copy()
        current_fitness = self.evaluator.evaluate(current)
        for _ in range(self.steps):
            candidate = current.copy()
            idx = np.random.randint(len(candidate))
            candidate[idx] ^= 1
            candidate_fitness = self.evaluator.evaluate(candidate)
            delta = candidate_fitness - current_fitness
            if delta > 0 or np.random.rand() < np.exp(delta / self.temperature):
                current = candidate
                current_fitness = candidate_fitness
            self.temperature *= self.cooling_rate
        return current
