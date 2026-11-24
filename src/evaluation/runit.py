
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score
from sklearn.linear_model import LogisticRegression

from data.runit import main as load_data
from rbn.network import RandomBooleanNetwork
from rbn.network import DataDrivenRBN
from rbn.simulator import RBNSimulator
from rbn.attractor import AttractorAnalyzer
from rbn.features import AttractorFeatureExtractor
from rbn.features import CombinedFeatureExtractor
from .classifier import RBNAttractorClassifier
from .comparison import ModelComparison
from .noise_robustness import NoiseRobustnessEvaluator
from utils.graphs import save_confusion_matrices,save_roc_data,save_feature_importance

def main():
    load_data()

    print("Loading processed data with selected features")
    features_dir=Path("data/processed/features")
    if features_dir.exists():
        try:
            X_train_binary=np.load(features_dir/"train_data_selected.npy")
            X_test_binary=np.load(features_dir/"test_data_selected.npy")
            print("  ✓ Loaded selected features")
        except FileNotFoundError:
            print("  Selected features not found, using full dataset")
            X_train_binary=np.load("data/processed/train_data_binary.npy")
            X_test_binary=np.load("data/processed/test_data_binary.npy")
    else:
        print("  Features directory not found, using full dataset")
        X_train_binary=np.load("data/processed/train_data_binary.npy")
        X_test_binary=np.load("data/processed/test_data_binary.npy")

    y_train=np.load("data/processed/train_labels.npy")
    y_test=np.load("data/processed/test_labels.npy")

    print(f"\nData shapes:")
    print(f"  Train: {X_train_binary.shape}")
    print(f"  Test:  {X_test_binary.shape}")
    print(f"\nClass distribution:")
    print(f"  Train: {np.bincount(y_train.astype(int))}")
    print(f"  Test:  {np.bincount(y_test.astype(int))}")

    print("Configuring Random Boolean Network...")
    n_nodes=20
    k=3
    print(f"  Network size (N): {n_nodes}")
    print(f"  Connectivity (K): {k}")

    network=DataDrivenRBN(n_nodes=n_nodes,k=3,random_state=42)
    print("  Learning Boolean rules from training data...")
    network.fit(X_train_binary,y_train)

    print("Creating RBN Attractor Classifier...")
    base_classifier=LogisticRegression(class_weight="balanced",max_iter=1000,C=1.0,random_state=42)
    rbn_classifier=RBNAttractorClassifier(
        network=network,
        base_classifier=base_classifier,
        handle_imbalance=True,
        decision_threshold=None
    )
    rbn_classifier.feature_extractor=CombinedFeatureExtractor()
    print("Using CombinedFeatureExtractor (12 features)")

    print("Training RBN classifier...")
    rbn_classifier.fit(X_train_binary,y_train)

    print("Optimizing decision threshold...")
    rbn_classifier.optimize_threshold(X_train_binary,y_train,metric="f1")

    print("Evaluating RBN classifier on test set...")
    y_pred_rbn=rbn_classifier.predict(X_test_binary)
    y_proba=rbn_classifier.predict_proba(X_test_binary)

    print("\nClassification Report:")
    print(classification_report(y_test,y_pred_rbn,zero_division=0))

    print("\nPrediction distribution:")
    pred_counts=np.bincount(y_pred_rbn.astype(int))
    print(f"  Class 0: {pred_counts[0]} predictions")
    print(f"  Class 1: {pred_counts[1] if len(pred_counts)>1 else 0} predictions")

    print("Training baseline Random Forest...")
    baseline_classifier=RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    baseline_classifier.fit(X_train_binary,y_train)
    y_pred_baseline=baseline_classifier.predict(X_test_binary)
    y_proba_baseline=baseline_classifier.predict_proba(X_test_binary)

    print("Comparing classifiers...")
    comparison_results=ModelComparison.compare(
        rbn_classifier,
        baseline_classifier,
        X_test_binary,
        y_test,
        label_a="RBN Classifier",
        label_b="Random Forest"
    )

    for label,metrics in comparison_results.items():
        print(f"\n{label}:")
        print(f"  Accuracy:         {metrics['accuracy']:.4f}")
        print(f"  F1 Score:         {metrics['f1_score']:.4f}")
        print(f"  ROC AUC:          {metrics.get('roc_auc','N/A')}")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    cm_rbn=confusion_matrix(y_test,y_pred_rbn)
    cm_random_forest=confusion_matrix(y_test,y_pred_baseline)
    save_confusion_matrices(cm_rbn,cm_random_forest)

    rbn_ROC_AUC=roc_auc_score(y_test,y_proba[:,1])
    rf_ROC_AUC=roc_auc_score(y_test,y_proba_baseline[:,1])
    fpr_rbn,tpr_rbn,_=roc_curve(y_test,y_proba[:,1])
    fpr_rf,tpr_rf,_=roc_curve(y_test,y_proba_baseline[:,1])

    save_roc_data(fpr_rbn,tpr_rbn,rbn_ROC_AUC,model_name="ROC_rbn")
    save_roc_data(fpr_rf,tpr_rf,rf_ROC_AUC,model_name="ROC_rf")

    if hasattr(baseline_classifier,"feature_importances_"):
        importance_rf=baseline_classifier.feature_importances_
        feature_names_path=Path("data/processed/feature_names.txt")
        selected_indices_path=Path("data/processed/features/selected_features.npy")
        if feature_names_path.exists() and selected_indices_path.exists():
            with open(feature_names_path,"r") as f:
                all_features=[line.strip() for line in f.readlines() if line.strip()]
            selected_indices=np.load(selected_indices_path)
            features_rf=[all_features[i] for i in selected_indices]
            print(f"Loaded {len(features_rf)} selected feature names from {len(all_features)} total")
            print(f"First 5 selected: {features_rf[:5]}")
            if len(features_rf)!=len(importance_rf):
                print(f"Warning: Mismatch - {len(features_rf)} names vs {len(importance_rf)} importances")
                features_rf=[f"Feature_{i}" for i in range(len(importance_rf))]
        else:
            print(f"  Could not find feature names or selected indices")
            features_rf=[f"Feature_{i}" for i in range(len(importance_rf))]
    else:
        importance_rf=None
        features_rf=None

    if hasattr(rbn_classifier.classifier,"coef_"):
        importance_rbn=np.abs(rbn_classifier.classifier.coef_[0])
        features_rbn=rbn_classifier.feature_extractor.get_feature_names()
    else:
        importance_rbn=None
        features_rbn=None

    if importance_rf is not None:
        save_feature_importance(features_rf,importance_rf,"Feature Importance Random Forest")
    if importance_rbn is not None:
        save_feature_importance(features_rbn,importance_rbn,"Feature Importance RBN")

    print("Evaluating noise robustness...")
    noise_robustness_evaluator=NoiseRobustnessEvaluator(rbn_classifier)
    noise_levels=[0.0,0.05,0.1,0.2]
    noise_robustness_results=noise_robustness_evaluator.evaluate(X_test_binary,y_test,noise_levels)

    print("\nNoise Robustness Results:")
    for noise,metrics in noise_robustness_results.items():
        print(f"\nNoise level: {noise:.2f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")

    print("Evaluation complete")

if __name__=="__main__":
    main()
