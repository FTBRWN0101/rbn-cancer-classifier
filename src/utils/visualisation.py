import os
os.environ["MPLBACKEND"] = "Agg"  
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "text.usetex": False,
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "backend": "Agg"
})

def plot_curve(x, y, title):
    """Plot a simple curve, checking that MatPlotLib is working properly."""
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "curve.png"
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker="o", linestyle="-", linewidth=2, markersize=6)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def plot_confusion_matrix(cm, labels, title, output_filename):
    """Plot confusion matrix.
    
    Parameters:
    cm : np.ndarray: Confusion matrix
    labels : list: Class labels
    title : str: Plot title
    output_filename : str: File name of output file (png file)"""
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot")


def plot_roc_curve(fpr, tpr, auc_score, title="ROC Curve", output_filename="roc_curve",):
    """Plot ROC curve.
    
    Parameters:
    fpr : np.ndarray: False positive rates
    tpr : np.ndarray: True positive rates
    auc_score : float: Area under curve
    title : str: Plot title
    """
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC curve (AUC = {auc_score:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: ROC_curve.png")


def plot_feature_importance(features, importance, title="Feature Importance", output_filename = "feature_importance"):
    """Plot feature importance as bar chart.
    
    Parameters:
    features : list: Feature names
    importance : np.ndarray: Importance scores
    title : str: Plot title"""
    output_dir = Path("results/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    indices = np.argsort(importance)[::-1]
    sorted_features = [features[i] for i in indices]
    sorted_importance = importance[indices]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(sorted_importance)), sorted_importance, align="center")
    plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha="right")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: feature_importance.png")