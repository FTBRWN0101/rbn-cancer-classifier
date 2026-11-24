# src/feature_selection/wrapper.py
import numpy as np
from typing import Callable, Optional
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.metrics import matthews_corrcoef

class FeatureSelectionWrapper:
    """
    Original wrapper-based feature selection.
    """
    def __init__(self, scoring_function: Callable):
        self.scoring_function = scoring_function

    def select(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        scores = self.scoring_function(X, y)
        ranked = np.argsort(scores)[::-1]
        return ranked[:k]

class ImprovedFeatureSelectionWrapper:
    """
    Wrapper for feature selection with class-aware methods.
    """
    
    def __init__(self, method: str = "combined"):
        """
        Args: method: One of 'mutual_info', 'chi2', 'fisher', 'differential', 'combined'
        """
        self.method = method
        self.selected_features_ = None
        self.scores_ = None
    
    def select(self, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """
        Select top k features.
        
        Args:, X: Binary feature matrix, y: Labels, k: Number of features to select
        Returns: Indices of selected features"""
        from .importance import (mutual_information_importance,chi_squared_importance,fisher_score_importance,differential_expression_importance,select_features_with_threshold)
        if self.method == "mutual_info":
            scores = mutual_information_importance(X, y)
        elif self.method == "chi2":
            scores = chi_squared_importance(X, y)
        elif self.method == "fisher":
            scores = fisher_score_importance(X, y)
        elif self.method == "differential":
            scores = differential_expression_importance(X, y)
        elif self.method == "combined":
            selected = select_features_with_threshold(X, y, k)
            self.selected_features_ = selected
            return selected
        else:
            raise ValueError(f"Unknown method: {self.method}")
        self.scores_ = scores
        self.selected_features_ = np.argsort(scores)[::-1][:k]
        return self.selected_features_
    
    def get_feature_scores(self) -> np.ndarray:
        """Get importance scores for all features."""
        if self.scores_ is None:
            raise ValueError("Must call select() first")
        return self.scores_