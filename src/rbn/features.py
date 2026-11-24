# src/rbn/features.py
import numpy as np
from scipy.stats import entropy
from typing import List


class AttractorFeatureExtractor:
    """
    Extracts numerical features from RBN attractors for classification.
    Features include:
    - Attractor length (period)
    - Mean activity across attractor
    - Entropy of attractor states
    - Stability metrics
    - State diversity measures
    """

    def __init__(self):
        """Initialize feature extractor."""
        self.n_features = 8
    
    @staticmethod
    def length(attractor: List[np.ndarray]) -> int:
        """
        Length of attractor cycle.
        Point attractors have length 1 and longer cycles may indicate more complex dynamics.
        """
        return len(attractor) if attractor else 0

    @staticmethod
    def mean_activity(attractor: List[np.ndarray]) -> float:
        """
        Average fraction of ON nodes across attractor. Measures overall activation level.
        """
        if not attractor:
            return 0.0
        states = np.vstack(attractor)
        return float(states.mean())
    
    @staticmethod
    def std_activity(attractor: List[np.ndarray]) -> float:
        """
        Standard deviation of activity across attractor states. High values indicate variable activation patterns.
        """
        if not attractor or len(attractor) == 1:
            return 0.0
        states = np.vstack(attractor)
        #Std across time for each node, then avg
        return float(states.std(axis=0).mean())

    @staticmethod
    def entropy(attractor: List[np.ndarray]) -> float:
        """Shannon entropy of node activation probabilities.
        High entropy = more disordered/chaotic attractor, Low entropy = more ordered attractor."""
        if not attractor:
            return 0.0
        states = np.vstack(attractor)
        probs = states.mean(axis=0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        node_entropies = -(probs * np.log2(probs) + 
                          (1 - probs) * np.log2(1 - probs))
        return float(node_entropies.mean())
    
    @staticmethod
    def hamming_diversity(attractor: List[np.ndarray]) -> float:
        """
        Average Hamming distance between consecutive states.
        Measures how much the attractor changes between steps.
        """
        if not attractor or len(attractor) == 1:
            return 0.0
        distances = []
        for i in range(len(attractor)):
            next_i = (i + 1) % len(attractor)
            hamming = np.sum(attractor[i] != attractor[next_i])
            distances.append(hamming)
        return float(np.mean(distances))
    
    @staticmethod
    def frozen_nodes(attractor: List[np.ndarray]) -> float:
        """
        Fraction of nodes that stay constant in attractor.

        Frozen nodes may represent stable regulatory decisions.
        """
        if not attractor:
            return 0.0
        states = np.vstack(attractor)
        #Node is frozen if it has the same val in all attractor state
        node_stds = states.std(axis=0)
        frozen = np.sum(node_stds == 0)
        return float(frozen / states.shape[1])
    
    @staticmethod
    def attractor_complexity(attractor: List[np.ndarray]) -> float:
        """
        Measure of attractor complexity based on unique states.
        
        More unique states = more complex attractor.
        """
        if not attractor:
            return 0.0
        unique_states = len(set(tuple(s) for s in attractor))
        return float(unique_states / len(attractor))
    
    @staticmethod
    def attractor_energy(attractor: List[np.ndarray]) -> float:
        """
        Pseudo-energy: sum of Hamming distances from all-zero state.
        
        Higher values = more activated states.
        """
        if not attractor:
            return 0.0
        states = np.vstack(attractor)
        energies = states.sum(axis=1)
        return float(energies.mean())

    def extract(self, attractor: List[np.ndarray]) -> np.ndarray:
        """
        Extract all features from attractor.
        Parameters
        attractor : List[np.ndarray]: List of binary state vectors forming the attractor
            
        Returns:
        np.ndarray: Feature vector of shape (n_features,)"""
        if not attractor:
            return np.zeros(self.n_features)
        
        features = np.array([
            self.length(attractor),
            self.mean_activity(attractor),
            self.std_activity(attractor),
            self.entropy(attractor),
            self.hamming_diversity(attractor),
            self.frozen_nodes(attractor),
            self.attractor_complexity(attractor),
            self.attractor_energy(attractor)
        ])
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of all extracted features."""
        return ['attractor_length','mean_activity','std_activity','entropy','hamming_diversity','frozen_nodes','complexity','energy']


class TransientFeatureExtractor:
    """Extracts features from the transient phase before reaching attractor.
    
    The transient can be informative about network dynamics"""
    
    def __init__(self):
        self.n_features = 4
    
    @staticmethod
    def transient_length(trajectory: List[np.ndarray],attractor_start: int) -> int:
        """Number of steps before reaching attractor.Long transients may indicate complex dynamics."""
        return attractor_start
    
    @staticmethod
    def transient_activity(trajectory: List[np.ndarray],attractor_start: int) -> float:
        """Mean activity during transient phase."""
        if attractor_start == 0:
            return 0.0
        transient = trajectory[:attractor_start]
        states = np.vstack(transient)
        return float(states.mean())
    
    @staticmethod
    def settling_stability(
        trajectory: List[np.ndarray],
        attractor_start: int
    ) -> float:
        """How quickly the network settles.
        
        Measured as mean Hamming distance between consecutive transient staes."""
        if attractor_start <= 1:
            return 0.0
        transient = trajectory[:attractor_start]
        distances = []
        for i in range(len(transient) - 1):
            dist = np.sum(transient[i] != transient[i + 1])
            distances.append(dist)
        return float(np.mean(distances)) if distances else 0.0
    
    @staticmethod
    def transient_entropy(
        trajectory: List[np.ndarray],
        attractor_start: int
    ) -> float:
        """Entropy during transient phase."""
        if attractor_start == 0:
            return 0.0
        transient = trajectory[:attractor_start]
        states = np.vstack(transient)
        probs = states.mean(axis=0)
        probs = np.clip(probs, 1e-10, 1 - 1e-10)
        node_entropies = -(probs * np.log2(probs) + 
                          (1 - probs) * np.log2(1 - probs))
        return float(node_entropies.mean())
    
    def extract(
        self,
        trajectory: List[np.ndarray],
        attractor_start: int
    ) -> np.ndarray:
        """Extract transient features."""
        return np.array([
            self.transient_length(trajectory, attractor_start),
            self.transient_activity(trajectory, attractor_start),
            self.settling_stability(trajectory, attractor_start),
            self.transient_entropy(trajectory, attractor_start)
        ])
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        return [
            'transient_length',
            'transient_activity',
            'settling_stability',
            'transient_entropy'
        ]


class CombinedFeatureExtractor:
    """
    Combines attractor and transient features for richer representation.
    """
    def __init__(self):
        self.attractor_extractor = AttractorFeatureExtractor()
        self.transient_extractor = TransientFeatureExtractor()
        self.n_features = (self.attractor_extractor.n_features + self.transient_extractor.n_features)
    
    def extract(self,trajectory: List[np.ndarray],attractor: List[np.ndarray],attractor_start: int) -> np.ndarray:
        """
        Extract both attractor and transient features.
        
        Parameters:
        trajectory : List[np.ndarray]: Full trajectory
        attractor : List[np.ndarray]: Attractor cycle
        attractor_start : int: Index where attractor begins
        Returns:
        np.ndarray: Combined feature vector
        """
        attractor_features = self.attractor_extractor.extract(attractor)
        transient_features = self.transient_extractor.extract(trajectory, attractor_start)
        return np.concatenate([attractor_features, transient_features])
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return (self.attractor_extractor.get_feature_names() + 
                self.transient_extractor.get_feature_names())