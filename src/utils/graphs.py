# src/utils/graphs.py
"""
Utilities for saving and loading graph data (confusion matrices, ROC curves, feature importance).
Separate module to avoid circular imports.
"""
import numpy as np
from pathlib import Path


def save_confusion_matrices(cm_rbn, cm_random_forest, output_dir="results/metrics"):
    """
    Save confusion matrices to disk for later visualization.
    Parameters:
    cm_rbn : np.ndarray: Confusion matrix from RBN classifier
    cm_random_forest : np.ndarray: Confusion matrix from Random Forest classifier
    output_dir : str or Path: Directory to save matrices
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "cm_rbn.npy", cm_rbn)
    np.save(output_dir / "cm_random_forest.npy", cm_random_forest)
    print(f"Saved confusion matrices to {output_dir}")


def load_confusion_matrices(input_dir="results/metrics"):
    """
    Load confusion matrices from disk.
    Parameters:
    input_dir : str or Path: Directory containing saved matrices
        
    Returns:
    tuple: (cm_rbn, cm_random_forest)
    """
    input_dir = Path(input_dir)
    cm_rbn = np.load(input_dir / "cm_rbn.npy")
    cm_random_forest = np.load(input_dir / "cm_random_forest.npy")
    return cm_rbn, cm_random_forest


def save_roc_data(fpr, tpr, auc_score, model_name="rbn", output_dir="results/metrics"):
    """
    Save ROC curve data to disk for later visualization.
    
    Parameters:
    fpr : np.ndarray: False positive rates
    tpr : np.ndarray: True positive rates
    auc_score : float: Area under curve
    model_name : str: Name of model (e.g., 'rbn', 'random_forest')
    output_dir : str or Path: Directory to save ROC data"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"fpr_{model_name}.npy", fpr)
    np.save(output_dir / f"tpr_{model_name}.npy", tpr)
    np.save(output_dir / f"auc_{model_name}.npy", np.array([auc_score]))
    print(f"Saved ROC data for {model_name} to {output_dir}")


def load_roc_data(model_name="rbn", input_dir="results/metrics"):
    """
    Load ROC curve data from disk.
    
    Parameters:
    model_name : str: Name of model (e.g., 'rbn', 'random_forest')
    input_dir : str or Path: Directory containing saved ROC data
    Returns:
    tuple: (fpr, tpr, auc_score)
    """
    input_dir = Path(input_dir)
    fpr = np.load(input_dir / f"fpr_{model_name}.npy")
    tpr = np.load(input_dir / f"tpr_{model_name}.npy")
    auc_score = np.load(input_dir / f"auc_{model_name}.npy")[0]
    return fpr, tpr, auc_score


def save_feature_importance(features, importance, model_name="rbn", output_dir="results/metrics"):
    """
    Save feature importance data to disk for later visualization.
    
    Parameters:
    features : list or np.ndarray: Feature names
    importance : np.ndarray: Importance scores
    model_name : str: Name of model (e.g., 'rbn', 'random_forest')
    output_dir : str or Path: Directory to save feature importance data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / f"feature_importance_{model_name}.npz",
             features=np.array(features, dtype=object),
             importance=importance)
    print(f"Saved feature importance for {model_name} to {output_dir}")


def load_feature_importance(model_name="rbn", input_dir="results/metrics"):
    """
    Load feature importance data from disk.
    
    Parameters:
    model_name : str: Name of model (e.g., 'rbn', 'random_forest')
    input_dir : str or Path: Directory containing saved feature importance data
    Returns:
    tuple: (features, importance)
    """
    input_dir = Path(input_dir)
    data = np.load(input_dir / f"feature_importance_{model_name}.npz", allow_pickle=True)
    features = data['features'].tolist()
    importance = data['importance']
    return features, importance