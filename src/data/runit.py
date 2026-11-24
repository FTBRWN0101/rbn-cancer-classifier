# src/data/runit.py
import numpy as np
from pathlib import Path

try:
    from .loader import load_ovarian_data
    from .preprocessor import DataPreprocessor, ImputationStrategy, NormalizationStrategy
    from .splitter import DataSplitter
    from .discretizer import GeneExpressionDiscretizer, DiscretizationMethod
except ImportError:
    from loader import load_ovarian_data
    from preprocessor import DataPreprocessor, ImputationStrategy, NormalizationStrategy
    from splitter import DataSplitter
    from discretizer import GeneExpressionDiscretizer, DiscretizationMethod

def main():
    """run data stage end to end"""
    print("Loading data")
    data, labels, feature_names = load_ovarian_data("data/raw_data/Ovarian.arff")

    print("\nPreprocessing data")
    preprocessor = DataPreprocessor(
        imputation_strategy=ImputationStrategy.MEAN,
        normalization_strategy=NormalizationStrategy.MIN_MAX,
        variance_threshold=1e-10,
    )
    normalized_data = preprocessor.fit_transform(data)

    print("\nSplitting data (70/30 with stratification)")
    splitter = DataSplitter(train_ratio=0.7, random_state=42, stratify=True)
    X_train, X_test, y_train, y_test = splitter.split(normalized_data, labels)

    print("\nDiscretizing data")
    print(f"  Continuous data range: [{X_train.min():.3f}, {X_train.max():.3f}]")

    discretizer = GeneExpressionDiscretizer(method=DiscretizationMethod.MEDIAN)
    X_train_binary = discretizer.fit_transform(X_train)
    X_test_binary = discretizer.transform(X_test)

    print("\nBinary data statistics:")
    print("  Training set:")
    print(f"    Shape: {X_train_binary.shape}")
    print(f"    ON genes (mean): {X_train_binary.mean(axis=0).mean():.3f}")
    print(f"    OFF genes (mean): {1 - X_train_binary.mean(axis=0).mean():.3f}")
    print("  Test set:")
    print(f"    Shape: {X_test_binary.shape}")
    print(f"    ON genes (mean): {X_test_binary.mean(axis=0).mean():.3f}")

    print("\nSaving processed data")
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "train_data_binary.npy", X_train_binary)
    np.save(output_dir / "test_data_binary.npy", X_test_binary)
    np.save(output_dir / "train_labels.npy", y_train)
    np.save(output_dir / "test_labels.npy", y_test)
    (output_dir / "feature_names.txt").write_text("\n".join(feature_names), encoding="utf-8")
    preprocessor.save_parameters(output_dir / "preprocessing_params.npz")
    discretizer.save_parameters(output_dir / "discretization_params.npz")
    splitter.save_indices(output_dir / "split_indices.npz")

    print("Data processing complete")

if __name__ == "__main__":
    main()
