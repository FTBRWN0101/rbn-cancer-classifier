import numpy as np
from typing import Optional
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.metrics import matthews_corrcoef

def variance_importance(X: np.ndarray, y=None) -> np.ndarray:
    """
    Variance-based gene importance."""
    return X.var(axis=0)

def mutual_information_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Mutual information between features and labels.
    Args:X: Binary feature matrix (n_samples, n_features)
         y: Labels (n_samples,)
    Returns: Importance scores for each feature"""
    return mutual_info_classif(X, y, discrete_features=True, random_state=42)

def chi_squared_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Chi-squared test for independence between features and labels.
    Args:X: Binary feature matrix (n_samples, n_features)
         y: Labels (n_samples,)
    Returns:Chi-squared statistics for each feature
    """
    scores, _ = chi2(X, y)
    return scores


def fisher_score_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fisher score: ratio of between-class to within-class variance.
    Args:X: Binary feature matrix (n_samples, n_features)
         y: Labels (n_samples,)
    Returns:Fisher scores for each feature
    """
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    classes = np.unique(y)
    n_c = {c: np.sum(y == c) for c in classes}
    mu_c = {c: X[y == c].mean(axis=0) for c in classes}
    mu = X.mean(axis=0)
    for i in range(n_features):
        numerator = sum(n_c[c] * (mu_c[c][i] - mu[i])**2 for c in classes)
        denominator = sum(np.sum((X[y == c, i] - mu_c[c][i])**2) for c in classes)
        scores[i] = numerator / (denominator + 1e-10)
    return scores


def differential_expression_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Measure how differently a gene is expressed between classes.
    Args:X: Binary feature matrix (n_samples, n_features)
         y: Labels (n_samples,)
    Returns: Differential expression scores
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("This method requires exactly 2 classes")
    mean_class_0 = X[y == classes[0]].mean(axis=0)
    mean_class_1 = X[y == classes[1]].mean(axis=0)
    return np.abs(mean_class_0 - mean_class_1)


def combined_importance(X: np.ndarray, y: np.ndarray,methods: Optional[list] = None) -> np.ndarray:
    """
    combine multiple importance measures using rank aggregation.    
    Args:X: Binary feature matrix
         y: Labels
    methods: list of importance functions to combine
    Returns: Combined importance scores
    """
    if methods is None:
        methods = [mutual_information_importance,fisher_score_importance,differential_expression_importance]
    # get ranks from each method
    ranks = []
    for method in methods:
        try:
            scores = method(X, y)
            rank = np.argsort(np.argsort(scores)[::-1])
            ranks.append(rank)
        except Exception as e:
            print(f"Warning: {method.__name__} failed: {e}")
            continue
    if not ranks:
        raise ValueError("All feature selection methods failed")
    combined_rank = np.mean(ranks, axis=0)
    return -combined_rank


def select_features_with_threshold(X: np.ndarray,y: np.ndarray,k: int,min_differential: float = 0.1) -> np.ndarray:
    """
    Select features using combined importance with minimum differential expression.
    Args:
        X: Binary feature matrix
        y: Labels
        k: Number of features to select
        min_differential: Minimum difference in expression between classes
    Returns:Indices of selected features
    """
    diff_expr = differential_expression_importance(X, y)
    valid_features = np.where(diff_expr >= min_differential)[0]
    if len(valid_features) == 0:
        print(f"Warning: No features with differential >= {min_differential}")
        print(f"Max differential found: {diff_expr.max():.3f}")
        valid_features = np.arange(X.shape[1])
    X_filtered = X[:, valid_features]
    scores = combined_importance(X_filtered, y)
    selected_indices = np.argsort(scores)[::-1][:k]
    return valid_features[selected_indices]
