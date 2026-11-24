import numpy as np
from typing import List
class RBNSimulator:
    """Simulates RBN dynamics from an initial states"""
    def __init__(self, network, max_steps: int = 200):
        self.network = network
        self.max_steps = max_steps

    def run(self, initial_state: np.ndarray) -> List[np.ndarray]:
        trajectory = []
        state = initial_state.copy()
        for _ in range(self.max_steps):
            trajectory.append(state.copy())
            state = self.network.step(state)
        return trajectory
