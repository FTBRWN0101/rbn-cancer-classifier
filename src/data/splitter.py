"""
train test splitting with stratification
main entry is DataSplitter.split
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

@dataclass
class SplitStatistics:
    """split statistics container"""
    n_train: int
    n_test: int
    train_ratio: float
    test_ratio: float
    class_distribution_train: dict
    class_distribution_test: dict

    def __str__(self) -> str:
        lines = [
            "Split Statistics:",
            f"  Training: {self.n_train} samples ({self.train_ratio*100:.1f}%)",
            f"  Testing: {self.n_test} samples ({self.test_ratio*100:.1f}%)",
            "\nClass Distribution:",
        ]
        for class_label in sorted(self.class_distribution_train.keys()):
            lines.append(f"  Class {class_label}:")
            lines.append(f"    Train: {self.class_distribution_train[class_label]}")
            lines.append(f"    Test: {self.class_distribution_test[class_label]}")
        return "\n".join(lines)

class DataSplitter:
    """70/30 split with optional stratification"""
    DEFAULT_TRAIN_RATIO = 0.7
    DEFAULT_RANDOM_STATE = 42

    def __init__(
        self,
        train_ratio: float = DEFAULT_TRAIN_RATIO,
        random_state: Optional[int] = DEFAULT_RANDOM_STATE,
        stratify: bool = True,
    ) -> None:
        self._validate_train_ratio(train_ratio)
        self.train_ratio = train_ratio
        self.random_state = random_state
        self.stratify = stratify
        self._train_indices: Optional[np.ndarray] = None
        self._test_indices: Optional[np.ndarray] = None
        self._statistics: Optional[SplitStatistics] = None

    @property
    def train_indices(self) -> Optional[np.ndarray]:
        return self._train_indices

    @property
    def test_indices(self) -> Optional[np.ndarray]:
        return self._test_indices

    @property
    def statistics(self) -> Optional[SplitStatistics]:
        return self._statistics

    @property
    def is_fitted(self) -> bool:
        return self._train_indices is not None

    def split(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        split data into train and test

        Returns:
            X_train, X_test, y_train, y_test
        """
        self._validate_inputs(data, labels)
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if self.stratify:
            indices_train, indices_test = self._stratified_split(labels)
        else:
            indices_train, indices_test = self._simple_split(len(data))
        self._train_indices = indices_train
        self._test_indices = indices_test
        X_train, X_test = data[indices_train], data[indices_test]
        y_train, y_test = labels[indices_train], labels[indices_test]
        self._statistics = self._compute_statistics(y_train, y_test)
        print(str(self._statistics))
        return X_train, X_test, y_train, y_test

    def apply_split(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """apply the stored indices to a new matrix"""
        if not self.is_fitted:
            raise ValueError("Must call split() before apply_split()")
        self._validate_inputs(data, labels)
        X_train = data[self._train_indices]
        X_test = data[self._test_indices]
        y_train = labels[self._train_indices]
        y_test = labels[self._test_indices]
        return X_train, X_test, y_train, y_test

    def save_indices(self, filepath: str | Path) -> None:
        """save split indices to npz"""
        if not self.is_fitted:
            raise ValueError("No split to save. Call split() first.")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            filepath,
            train_indices=self._train_indices,
            test_indices=self._test_indices,
            train_ratio=self.train_ratio,
            random_state=self.random_state,
            stratify=self.stratify,
        )

    def load_indices(self, filepath: str | Path) -> "DataSplitter":
        """load split indices from npz"""
        params = np.load(filepath, allow_pickle=True)
        self._train_indices = params["train_indices"]
        self._test_indices = params["test_indices"]
        self.train_ratio = float(params["train_ratio"])
        self.random_state = int(params["random_state"]) if params["random_state"] is not None else None
        self.stratify = bool(params["stratify"])
        return self

    def _simple_split(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.permutation(n_samples)
        n_train = int(n_samples * self.train_ratio)
        return indices[:n_train], indices[n_train:]

    def _stratified_split(self, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        train_indices = []
        test_indices = []
        for class_label in np.unique(labels):
            class_indices = np.where(labels == class_label)[0]
            np.random.shuffle(class_indices)
            n_train = int(len(class_indices) * self.train_ratio)
            train_indices.extend(class_indices[:n_train])
            test_indices.extend(class_indices[n_train:])
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)
        return train_indices, test_indices

    def _compute_statistics(self, y_train: np.ndarray, y_test: np.ndarray) -> SplitStatistics:
        n_total = len(y_train) + len(y_test)
        train_dist = {int(label): int(np.sum(y_train == label)) for label in np.unique(y_train)}
        test_dist = {int(label): int(np.sum(y_test == label)) for label in np.unique(y_test)}
        return SplitStatistics(
            n_train=len(y_train),
            n_test=len(y_test),
            train_ratio=len(y_train) / n_total,
            test_ratio=len(y_test) / n_total,
            class_distribution_train=train_dist,
            class_distribution_test=test_dist,
        )

    @staticmethod
    def _validate_train_ratio(train_ratio: float) -> None:
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")

    @staticmethod
    def _validate_inputs(data: np.ndarray, labels: np.ndarray) -> None:
        if len(data) != len(labels):
            raise ValueError(f"data and labels must have same length: {len(data)} != {len(labels)}")
        if len(data) == 0:
            raise ValueError("Cannot split empty dataset")
