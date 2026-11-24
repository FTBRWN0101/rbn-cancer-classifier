# src/feature_selection/runit.py
from pathlib import Path
import numpy as np
from data.runit import main as load_data
from .wrapper import ImprovedFeatureSelectionWrapper
from .importance import differential_expression_importance

def main():
    load_data()
    print("Loading processed data")
    data_dir = Path("data/processed")
    
    X_train_binary = np.load(data_dir / "train_data_binary.npy")
    X_test_binary = np.load(data_dir / "test_data_binary.npy")
    y_train = np.load(data_dir / "train_labels.npy")
    y_test = np.load(data_dir / "test_labels.npy")
    
    print(f"\nData loaded:")
    print(f"  Train: {X_train_binary.shape}, labels: {y_train.shape}")
    print(f"  Test: {X_test_binary.shape}, labels: {y_test.shape}")
    print(f"  Class distribution (train): {np.bincount(y_train.astype(int))}")
    print(f"  Class distribution (test): {np.bincount(y_test.astype(int))}")

    print("Feature selection using combined class-aware importance")
    
    k = 30 #can change, more is better discrimination.
    feature_selector = ImprovedFeatureSelectionWrapper(method="combined")
    selected_features = feature_selector.select(X_train_binary, y_train, k)
    print(f"\nSelected {len(selected_features)} features:")
    print(f"  Indices: {selected_features}")
    
    print("Differential expression analysis of selected features:")
    diff_expr = differential_expression_importance(X_train_binary, y_train)
    
    for i, feat_idx in enumerate(selected_features[:10]):  #top 10
        class_0_expr = X_train_binary[y_train == 0, feat_idx].mean()
        class_1_expr = X_train_binary[y_train == 1, feat_idx].mean()
        print(f"  Feature {feat_idx:4d}: "
              f"diff={diff_expr[feat_idx]:.3f}, "
              f"class0={class_0_expr:.3f}, "
              f"class1={class_1_expr:.3f}")

    X_train_selected = X_train_binary[:, selected_features]
    X_test_selected = X_test_binary[:, selected_features]

    print(f"Data after feature selection:")
    print(f"  Training set shape: {X_train_selected.shape}")
    print(f"  Test set shape: {X_test_selected.shape}")
    
    print("Class-specific expression patterns:")
    for class_label in [0, 1]:
        class_data = X_train_selected[y_train == class_label]
        print(f"  Class {int(class_label)} ({len(class_data)} samples):")
        print(f"    Mean ON rate: {class_data.mean():.3f}")
        print(f"    Std ON rate:  {class_data.std():.3f}")

    print("Saving selected features")    
    output_dir = Path("data/processed/features")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "train_data_selected.npy", X_train_selected)
    np.save(output_dir / "test_data_selected.npy", X_test_selected)
    np.save(output_dir / "selected_features.npy", selected_features)
    
    np.save(output_dir / "differential_expression_scores.npy", diff_expr)
    print(f"  Saved to: {output_dir}")
    print("Feature selection complete.")

if __name__ == "__main__":
    main()