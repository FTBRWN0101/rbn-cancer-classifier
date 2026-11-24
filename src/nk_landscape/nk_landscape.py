import numpy as np
from typing import List


class NKLandscape:
    """
    NK fitness landscape with binary genome representation.
    """

    def __init__(self, n: int, k: int, random_state: int | None = None):
        self.n = n
        self.k = k

        if random_state is not None:
            np.random.seed(random_state)

        self.interactions = self._generate_interactions()
        self.fitness_tables = self._generate_fitness_tables()

    def _generate_interactions(self) -> List[np.ndarray]:
        return [
            np.random.choice(
                [j for j in range(self.n) if j != i],
                self.k,
                replace=False
            )
            for i in range(self.n)
        ]

    def _generate_fitness_tables(self) -> List[np.ndarray]:
        return [
            np.random.rand(2 ** (self.k + 1))
            for _ in range(self.n)
        ]

    def fitness(self, genome: np.ndarray) -> float:
        total = 0.0

        for i in range(self.n):
            indices = [genome[i]] + genome[self.interactions[i]].tolist()
            idx = int("".join(map(str, indices)), 2)
            total += self.fitness_tables[i][idx]

        return total / self.n
