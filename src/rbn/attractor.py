import numpy as np
from typing import List, Tuple

class AttractorAnalyzer:
    """Identifies attractor cycles in RBN state trajectories.
    
    Enhanced to also track where the attractor begins (transient length)."""
    @staticmethod
    def find_attractor(
        trajectory: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Finds attractor cycle in trajectory.
        Parameters:
        trajectory : List[np.ndarray] that is a sequence of binary states
        Returns
        List[np.ndarray]: States forming the attractor cycle"""
        seen = {}
        for step, state in enumerate(trajectory):
            key = tuple(state.tolist())
            if key in seen:
                start = seen[key]
                return trajectory[start:step]
            seen[key] = step
        #no attractor found
        return []
    
    @staticmethod
    def find_attractor_with_transient(
        trajectory: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], int]:
        """
        Find attractor and return where it starts.
        
        Parameters:
        trajectory : List[np.ndarray] is a list of sequences of binary states
        Returns:
        attractor : List[np.ndarray] taht contains states forming the attractor cycle
        attractor_start : int, where Index where attractor begins (transient length)
        """
        seen = {}
        for step, state in enumerate(trajectory):
            key = tuple(state.tolist())
            if key in seen:
                start = seen[key]
                attractor = trajectory[start:step]
                return attractor, start
            seen[key] = step
        #no attractor found
        return [], len(trajectory)
    
    @staticmethod
    def basin_of_attraction_size(
        network,
        attractor: List[np.ndarray],
        n_samples: int = 100
    ) -> float:
        """
        Estimate basin of attraction size via random sampling.
        takes:
        network : RandomBooleanNetwork: Network type to simulate
        attractor : List[np.ndarray]: Known attractor
        n_samples : int: Number of random initial states to test
        Returns
        float: Estimated fraction of state space in this basim"""
        from .simulator import RBNSimulator
        simulator = RBNSimulator(network, max_steps=50)
        analyzer = AttractorAnalyzer()
        #Convert attractor to tuples
        attractor_set = set(tuple(s.tolist()) for s in attractor)
        in_basin = 0
        for _ in range(n_samples):
            initial = np.random.randint(0, 2, size=network.n_nodes)
            #run dynamics
            trajectory = simulator.run(initial)
            found_attractor = analyzer.find_attractor(trajectory)
            found_set = set(tuple(s.tolist()) for s in found_attractor)
            if found_set == attractor_set:
                in_basin += 1
        return in_basin / n_samples
    
    @staticmethod
    def attractor_stability(
        network,
        attractor: List[np.ndarray],
        noise_level: float = 0.01,
        n_trials: int = 10
    ) -> float:
        """
        Measure attractor stability under noise.
        
        Parameters:
        network : RandomBooleanNetwork: Network to simulate
        attractor : List[np.ndarray]: Attractor to test
        noise_level : float: Probability of flipping each bit
        n_trials : int: Number of noisy perturbations to test
        Returns
        float: Fraction of trials that return to same attractor
        """
        from .simulator import RBNSimulator
        simulator = RBNSimulator(network, max_steps=50)
        analyzer = AttractorAnalyzer()
        attractor_set = set(tuple(s.tolist()) for s in attractor)
        returns = 0
        for _ in range(n_trials):
            #start from random state
            start_state = attractor[np.random.randint(len(attractor))].copy()
            flip_mask = np.random.rand(len(start_state)) < noise_level
            start_state[flip_mask] = 1 - start_state[flip_mask]
            trajectory = simulator.run(start_state)
            found_attractor = analyzer.find_attractor(trajectory)
            found_set = set(tuple(s.tolist()) for s in found_attractor)
            if found_set == attractor_set:
                returns += 1
        return returns / n_trials