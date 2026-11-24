import os
os.environ['MPLBACKEND'] = 'Agg'
from .visualisation import(
    plot_curve, 
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance)
from .logging_utils import setup_logger
from .paths import ProjectPaths
from .validation import validate_binary_matrix, validate_shapes
from .graphs import(
    load_confusion_matrices,
    load_feature_importance,
    load_roc_data
)
import numpy as np

def main():
    """Test utility functions."""
    #logger
    print("\nSetting up logger.")
    logger = setup_logger("utils_logger")
    logger.info("Logger setup complete.")
    #project paths
    print("\nCreating project paths.")
    ProjectPaths.ensure_all()
    logger.info("Project paths created.")
    #validate data
    print("\nValidating data")
    data = np.random.randint(0, 2, size=(10, 5))
    labels = np.random.randint(0, 2, 10)
    validate_binary_matrix(data)
    validate_shapes(data, labels)
    logger.info("Data validation passed.")
    #example curve
    print("\nPlotting example curve.")
    x_eg = np.linspace(0, 10, 100)
    y_eg = np.sin(x_eg)
    plot_curve(x_eg, y_eg, "Example Curve")
    #confusion matrix
    print("\nPlotting confusion matrix.")
    cm_rbn, cm_rf = load_confusion_matrices()
    plot_confusion_matrix(cm_rbn, ["Cancer cell", "Normal cell"], "Comparison Matrix of RBN", "confusion_matrix_rbn")
    plot_confusion_matrix(cm_rf, ["Cancer cell", "Normal cell"], "Comparison Matrix of Random Forest", "confusion_matrix_random_forest")
    #ROC Curve
    print("\nPlotting ROC curve.")
    fpr_rbn, tpr_rbn, AUC_rbn = load_roc_data(model_name="ROC_rbn")
    fpr_rf, tpr_rf, AUC_rf = load_roc_data(model_name="ROC_rf")
    plot_roc_curve(fpr_rbn, tpr_rbn, AUC_rbn, "RBN ROC curve", "rbn_roc_curve")
    plot_roc_curve(fpr_rf, tpr_rf, AUC_rf, "RF ROC curve", "rf_roc_curve")
    #feature importance
    print("\nPlotting feature importance.")
    try:
        features_rf, importance_rf = load_feature_importance(model_name="Feature Importance Random Forest")
        plot_feature_importance(features_rf, importance_rf, title="Feature Importance Random Forest", output_filename="rf_feature_importance.png")
    except FileNotFoundError:
        print("Random Forest feature importance not found!")
    try:
        features_rbn, importance_rbn = load_feature_importance(model_name="Feature Importance RBN")
        plot_feature_importance(features_rbn, importance_rbn,title="Feature Importance RBN",output_filename="rbn_feature_importance.png")
    except FileNotFoundError:
        print("RBN feature importance not found!")
    print("\nUtils testing complete!")


if __name__ == "__main__":
    main()